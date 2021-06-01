import argparse
import glob
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, "../")
sys.path.insert(0, "../dpr")

import torch
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer, scoring
from pytorch_lightning.loggers import WandbLogger

from utils.tokenizers import SimpleTokenizer
from data.qa_validation import exact_match_score, has_answer
from lightning_base import BaseTransformer, generic_train, get_linear_schedule_with_warmup
from utils_gen import SummarizationDataset, choose_gpu, label_smoothed_nll_loss, reweight_labels, \
    freeze_params
from conf import add_generic_args, add_model_specific_args


def calculate_rouge(output_lns, reference_lns, score_path=None):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    if score_path is not None:
        score_file = Path(score_path).open("w")
        score_file.write(
            "ROUGE_1: \n{} \n\n ROUGE_2: \n{} \n\n ROUGE_L: \n{} \n\n".format(
                result["rouge1"], result["rouge2"], result["rougeL"]
            )
        )
    return result


class SummarizationTrainer(BaseTransformer):
    mode = "language-modeling"

    def __init__(self, hparams):
        super().__init__(hparams, num_labels=None, mode=self.mode)
        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            max_target_length=self.hparams.max_target_length,
        )
        self.all_answers = {}
        if os.path.exists(os.path.join(self.hparams.data_dir, "val.target.json")):
            self.all_answers['val'] = json.load(open(os.path.join(self.hparams.data_dir, "val.target.json")))
        if os.path.exists(os.path.join(self.hparams.data_dir, "test.target.json")):
            self.all_answers['test'] = json.load(open(os.path.join(self.hparams.data_dir, "test.target.json")))
        self.simple_tokenizer = SimpleTokenizer()

        if self.hparams.freeze_embeds:
            self.freeze_embeds()
        if self.hparams.freeze_encoder:
            freeze_params(self.model.get_encoder())
        if self.hparams.freeze_decoder:
            freeze_params(self.model.model.decoder)
        par_finetuned = []
        for name, par in self.model.named_parameters():
            if par.requires_grad:
                par_finetuned.append(name)
        if len(par_finetuned) < 10:
            print('\n >>>[warning]requires_grad:', par_finetuned)

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        except AttributeError:
            freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                freeze_params(d.embed_tokens)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def _step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = batch["source_ids"], batch["source_mask"], batch["target_ids"]

        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()  # next tokens (different from newest HF implementation)
        lm_labels[y[:, 1:] == pad_token_id] = -100

        # NB. kw_labels is not used in this project
        kw_labels = None
        if 'kw_labels' in batch:
            # NB [:, 1:] is to match with lm_labels
            kw_labels = batch["kw_labels"][:, 1:]
        # outputs = self(source_ids, attention_mask=source_mask, decoder_input_ids=y_ids, lm_labels=lm_labels, )
        # loss = outputs[0]
        # add use_cache=False when HF==3
        outputs = self(source_ids, attention_mask=source_mask, decoder_input_ids=y_ids, use_cache=False)
        lm_logits = outputs[0]
        if self.hparams.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            if kw_labels is None or self.hparams.label_decay == 1:
                ce_loss_fct = torch.nn.CrossEntropyLoss()
                loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), lm_labels.view(-1))
            else:
                ce_loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), lm_labels.view(-1))
                token_weights = reweight_labels(kw_labels, self.hparams.label_decay)
                pad_ct = torch.where(lm_labels.view(-1) == -100)[0].shape[0]
                total_ct = lm_labels.view(-1).shape[0]
                loss = torch.sum(loss * token_weights.view(-1)) / (total_ct - pad_ct)
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, lm_labels, self.hparams.label_smoothing
            )

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        preds, target = self.decode(batch)
        loss = self._step(batch)
        return {"val_loss": loss, "preds": preds, "target": target}

    def validation_epoch_end(self, outputs):
        rouge_res = self.write_results(outputs, 'val')
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        res = {'epoch': self.current_epoch, "val_loss": avg_loss, **rouge_res}
        return {'log': res}

    def decode(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = SummarizationDataset.trim_seq2seq_batch(batch, pad_token_id)
        generated_ids = self.model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            num_beams=1,
            max_length=self.hparams.max_target_length,
            # repetition_penalty=2.5,
            # length_penalty=1.0,
            early_stopping=True
        )

        preds = [
            self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for g in generated_ids
        ]
        target = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False) for t in y]
        return preds, target

    def calc_EM(self, pred_answers, all_answers):
        n_EM = 0
        for pred, answers in zip(pred_answers, all_answers):
            if any(exact_match_score(pred, g) for g in answers):
                n_EM += 1
        EM = n_EM / len(all_answers) * 100
        return EM

    def calc_acc(self, pred_answers, all_answers):
        n_acc = 0
        for pred, answers in zip(pred_answers, all_answers):
            if has_answer(answers=answers, text=pred, tokenizer=self.simple_tokenizer, match_type='string'):
                n_acc += 1
        acc = n_acc / len(pred_answers) * 100
        return acc

    def write_results(self, outputs, mode):
        targets_file = Path(self.hparams.output_dir) / f"{mode}_targets.txt"
        predictions_file = Path(self.hparams.output_dir) / f"{mode}_predictions-{self.current_epoch}.txt"
        p_writer = open(predictions_file, "w+")
        t_writer = open(targets_file, "w+")
        # score_file = Path(self.hparams.output_dir) / f"{mode}_ROUGE-{self.current_epoch}.txt"
        # write predictions and targets for later rouge evaluation.
        retry = 5
        while retry:
            retry -= 1
            try:
                for output_batch in outputs:
                    p_writer.writelines(s + "\n" for s in output_batch["preds"])
                    t_writer.writelines(s + "\n" for s in output_batch["target"])
                p_writer.close()
                t_writer.close()
                break
            except OSError as e:
                # Bad address, retry
                print(e)

        pred_l = []
        tgt_l = []
        for output_batch in outputs:
            pred_l.extend(output_batch['preds'])
            tgt_l.extend(output_batch['target'])
        EM = -1
        acc = -1
        if mode in self.all_answers:
            if self.hparams.max_target_length > 50:  # generate context
                acc = self.calc_acc(pred_l, self.all_answers[mode])
            else:  # generate answer
                EM = self.calc_EM(pred_l, self.all_answers[mode])

        # output_lns = [x.rstrip() for x in open(predictions_file).readlines()]
        # reference_lns = [x.rstrip() for x in open(targets_file).readlines()]
        # result = calculate_rouge(output_lns, reference_lns)
        result = calculate_rouge(pred_l, tgt_l)
        return_dict = {f'{mode}-ROUGE-1': torch.tensor(result["rouge1"].mid.fmeasure),
                       f'{mode}-ROUGE-2': torch.tensor(result["rouge2"].mid.fmeasure),
                       f'{mode}-ROUGE-L': torch.tensor(result["rougeL"].mid.fmeasure), }
        if EM != -1:
            return_dict[f'{mode}-EM'] = torch.tensor(EM)
        if acc != -1:
            return_dict[f'{mode}-acc'] = torch.tensor(acc)
        return return_dict

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_end(self, outputs):
        rouge_res = self.write_results(outputs, 'test')
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        res = {'epoch': self.current_epoch, "test_loss": avg_loss, **rouge_res}
        return {'log': res}

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False,
                       num_workers: int = 0) -> DataLoader:
        dataset = SummarizationDataset(self.tokenizer, type_path=type_path, **self.dataset_kwargs)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn, shuffle=shuffle,
                                num_workers=num_workers)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True, num_workers=4)
        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self) -> DataLoader:
        try:
            return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size, num_workers=4)
        except Exception as e:
            print(e)
            return self.get_dataloader("train", batch_size=self.hparams.eval_batch_size, num_workers=4)

    def test_dataloader(self) -> DataLoader:
        try:
            return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size, num_workers=4)
        except Exception as e:
            print(e)
            return self.get_dataloader("train", batch_size=self.hparams.eval_batch_size, num_workers=4)


def main(args):
    model = SummarizationTrainer(args)

    cp_file = None
    if args.do_train:
        if args.load_ckpt_name is not None:
            # load state_dict only
            print(f'load weight only from {args.load_ckpt_name} ...')
            ckpt = torch.load(args.load_ckpt_name)
            ckpt['state_dict'] = dict(ckpt['state_dict'])
            for k in list(ckpt['state_dict']):
                ckpt['state_dict'][k[6:]] = ckpt['state_dict'].pop(k)
            model.model.load_state_dict(ckpt['state_dict'], strict=False)
            del ckpt  # del ckpt otherwise OOM
            torch.cuda.empty_cache()
        else:
            if (Path(args.output_dir) / 'checkpointlast.ckpt').exists():
                cp_file = str(Path(args.output_dir) / 'checkpointlast.ckpt')
            else:
                cp_files = glob.glob(os.path.join(args.output_dir, '*.ckpt'))
                if len(cp_files) > 0:
                    cp_file = sorted(cp_files, key=lambda x: int(x.split('=')[-1][:-5]), reverse=True)[0]
            if cp_file is not None:
                print(f'resume training from {cp_file} ...')

    # logger = WandbLogger(name=args.remark, project="GAR")  # use wandb logger if you want
    logger = True
    trainer = generic_train(model, args, logger, resume_cp_file=cp_file, )

    if args.do_train:
        # (1) load the best checkpoint automatically (lightning tracks this for you)
        trainer.test()
    elif args.do_predict:
        # checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpointepoch=*.ckpt"), recursive=True)))
        print('checkpoint:', args.ckpt_name)
        model = model.load_from_checkpoint(str(args.ckpt_name))
        trainer.test(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    print('output_dir=', args.output_dir)
    if 'workspace' in str(args.data_dir) or 'mount' in str(args.data_dir):
        choose_gpu(min_gpu_memory=args.min_gpu_memory, retry=True)
    else:
        args.n_gpu = args.gpus = torch.cuda.device_count()

    main(args)

import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm

from transformers import AutoModelWithLMHead, AutoTokenizer
from transformers.configuration_auto import AutoConfig

from conf import setup_task
from train_generator import calculate_rouge
from utils_gen import choose_gpu


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def generate_summaries(
        examples: list, out_file: str, model_name: str, batch_size: int = 8, device: str = 'cuda',
        model_ckpt: str = None, max_tgt_len: int = 1024, max_src_length: int = 1024,
):
    fout = Path(out_file).open("w", encoding="utf-8")
    cfg = AutoConfig.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name, config=cfg).to(device)
    if model_ckpt is not None:
        print('Loading checkpoint from %s' % model_ckpt)
        ckpt = torch.load(model_ckpt)
        print(ckpt.keys())
        # TODO temporary workaround of loading pl.Trainer
        ckpt['state_dict'] = dict(ckpt['state_dict'])
        for k in list(ckpt['state_dict']):
            ckpt['state_dict'][k[6:]] = ckpt['state_dict'].pop(k)
        model.load_state_dict(ckpt['state_dict'])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # update config with summarization specific params
    # task_specific_params = model.config.task_specific_params
    # if task_specific_params is not None:
    #     model.config.update(task_specific_params.get("summarization", {}))
    for batch in tqdm(list(chunks(examples, batch_size))):
        if "t5" in model_name:
            batch = [model.config.prefix + text for text in batch]
        # NB. use this when using transformers==2.11.0
        dct = tokenizer.batch_encode_plus(batch, max_length=max_src_length, return_tensors="pt",
                                          pad_to_max_length=True).to(device)
        # NB. use this when using transformers==3.1.0
        # dct = tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(device)

        sampling = False
        if sampling:
            dataset = 'trivia'
            if dataset == 'nq':
                temperature = 5
                top_p = .5
            else:
                temperature = 2
                top_p = .5
            summaries = model.generate(num_beams=1,
                                       max_length=max_tgt_len,
                                       do_sample=True,
                                       temperature=temperature,
                                       top_p=top_p,
                                       num_return_sequences=10,
                                       early_stopping=True, **dct)
        else:
            summaries = model.generate(num_beams=1,
                                       max_length=max_tgt_len,
                                       early_stopping=True, **dct)
        dec = tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for hypothesis in dec:
            fout.write(hypothesis + "\n")
            fout.flush()


if __name__ == "__main__":
    mode, output_path, data_path, MAX_SRC_LENGTH, MAX_TGT_LENGTH, remark, bs, bs_eval, ckpt_name = setup_task()
    output_path = data_path
    split = 'test'

    parser = argparse.ArgumentParser()
    parser.add_argument("--remark", default=remark, type=str)
    parser.add_argument("--model_ckpt", type=str, default=ckpt_name)
    parser.add_argument("--input_path", type=str, default=data_path / f'{split}.source')
    parser.add_argument("--reference_path", type=str, required=False, default=data_path / f'{split}.target')
    parser.add_argument("--max_source_length", default=MAX_SRC_LENGTH, type=int)
    parser.add_argument("--max_target_length", default=MAX_TGT_LENGTH, type=int)
    parser.add_argument(
        "--output_path", type=str, default=output_path / f'gen-{split}-{remark}.txt',
        help="where to save summaries",
    )
    parser.add_argument(
        "--score_path", type=str, required=False, default=output_path / f'ROUGE-{remark}.txt',
        help="where to save the rouge score",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/bart-large",
        help="like bart-large-cnn,'t5-small', 't5-base', 't5-large', 't5-3b', 't5-11b",
    )
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--min_gpu_memory", type=int, default=7500)
    parser.add_argument("--bs", type=int, default=bs_eval, required=False, help="batch size")
    args = parser.parse_args()
    examples = [" " + x.rstrip() if "t5" in args.model_name else x.rstrip() for x in open(args.input_path).readlines()]

    choose_gpu(retry=True, min_gpu_memory=15000)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(args.output_path, args.input_path)
    if Path(args.output_path).exists() and 'del' not in args.output_path:
        print(args.output_path, 'exists! exiting...')
        exit()

    generate_summaries(examples, args.output_path, args.model_name, batch_size=args.bs, device=args.device,
                       model_ckpt=args.model_ckpt, max_tgt_len=args.max_target_length,
                       max_src_length=args.max_source_length)
    if args.score_path is not None:
        output_lns = [x.rstrip() for x in open(args.output_path).readlines()]
        reference_lns = [x.rstrip() for x in open(args.reference_path).readlines()]
        calculate_rouge(output_lns, reference_lns, args.score_path)

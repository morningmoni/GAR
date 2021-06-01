import json
import os
import pickle
import random
import subprocess
import time
from multiprocessing import get_context, Pool
from time import sleep
from datetime import datetime
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


# from transformers.tokenization_utils import trim_batch

def trim_batch(
        input_ids,
        pad_token_id,
        attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


def chunks(l, n):
    n = len(l) // n
    n = max(1, n)
    return (l[i:i + n] for i in range(0, len(l), n))


def multi_runs(f, para, f_combine=None, n=10):
    with get_context("spawn").Pool(n) as pool:
        # with Pool(n) as pool:
        res = pool.map(f, para)
        if f_combine is not None:
            res = f_combine(res)
        return res


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def read_retrieval_N(fname, data, pad_k=0):
    # use this func to avoid zero retreived mismatch

    retrieved_docs = [[] for _ in range(len(data))]
    with open(fname, encoding='utf8') as f:
        for line in tqdm(f):
            data = line.strip().split('\t')
            cur = int(data[0])
            retrieved_docs[cur].append(data[1])
    if pad_k > 0:
        ct = 0
        for i in range(len(retrieved_docs)):
            if len(retrieved_docs[i]) < pad_k:
                retrieved_docs[i].extend(['1'] * (pad_k - len(retrieved_docs[i])))
                ct += 1
        if ct > 0:
            print(f'insert random passage for {ct} samples. otherwise reader may crash')
    return retrieved_docs


def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free,utilization.gpu',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_info = [eval(x) for x in result.strip().split('\n')]
    gpu_info = dict(zip(range(len(gpu_info)), gpu_info))
    sorted_gpu_info = sorted(gpu_info.items(), key=lambda kv: kv[1][0], reverse=True)
    sorted_gpu_info = sorted(sorted_gpu_info, key=lambda kv: kv[1][1])
    return sorted_gpu_info


def choose_gpu(n_gpus=1, min_gpu_memory=9000, retry=False, sleep_time=30):
    start_time = time.time()
    sorted_gpu_info = get_gpu_memory_map()
    print(f'gpu_id, (mem_left, util): {sorted_gpu_info}')
    while True:
        gpus = []
        for gpu_id, (mem_left, util) in sorted_gpu_info:
            if mem_left >= min_gpu_memory:
                gpus.append(gpu_id)
                print('use gpu:{} with {} MB left, util {}%'.format(gpu_id, mem_left, util))
            if len(gpus) == n_gpus:
                print('max num of gpus reached.')
                break
        if len(gpus) == 0:
            if retry:
                print(f'[{datetime.now().strftime("%H:%M:%S")}'
                      f' waited {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}]'
                      f' no gpu has memory >= {min_gpu_memory} MB, sleep {sleep_time}s...', end='\r')
                time.sleep(sleep_time)
            else:
                print(f'no gpu has memory >= {min_gpu_memory} MB, exiting...')
                exit()
        else:
            break
        sorted_gpu_info = get_gpu_memory_map()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    visible_gpus = ','.join([str(gpu_id) for gpu_id in gpus])
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus


def encode_file(tokenizer, data_path, max_length, pad_to_max_length=True, return_tensors="pt"):
    examples = []
    with open(data_path, encoding='utf8') as f:
        for text in tqdm(f.readlines()):
            tokenized = tokenizer.batch_encode_plus(
                [text], max_length=max_length, pad_to_max_length=pad_to_max_length, return_tensors=return_tensors,
            )
            examples.append(tokenized)
    return examples


# workaround to pickle tokenization results
# https://github.com/huggingface/transformers/issues/4327
from transformers.tokenization_utils import BatchEncoding


def red(self):
    return BatchEncoding, (self.data,)


BatchEncoding.__reduce__ = red


class SummarizationDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            data_dir="./cnn-dailymail/cnn_dm/",
            type_path="train",
            max_source_length=1024,
            max_target_length=56,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.type_path = type_path
        if 'bart' in str(tokenizer):
            suffix = ''
        elif 't5' in str(tokenizer):
            suffix = '.t5'
        else:
            raise NotImplementedError

        self.concat_source = True  # NB. if not concat_src, can dynamic shuffle from top-100
        self.generate_relevance = False  # NB. if True, target is YES/NO instead of answers

        if self.concat_source:
            if os.path.exists(os.path.join(data_dir, type_path + f".source.processed{suffix}")):
                print(f'loading from {type_path}.processed{suffix} (pkl)... make sure data is what you need')
                self.source = pickle.load(open(os.path.join(data_dir, type_path + f".source.processed{suffix}"), 'rb'))
                self.target = pickle.load(open(os.path.join(data_dir, type_path + f".target.processed{suffix}"), 'rb'))
            else:
                # self.source = encode_file(tokenizer, os.path.join(data_dir, type_path + ".source.rerank_ext_vote_top5"), max_source_length)
                self.source = encode_file(tokenizer, os.path.join(data_dir, type_path + ".source"), max_source_length)
                self.target = encode_file(tokenizer, os.path.join(data_dir, type_path + ".target"), max_target_length)
                pickle.dump(self.source, open(os.path.join(data_dir, type_path + f".source.processed{suffix}"), 'wb'))
                pickle.dump(self.target, open(os.path.join(data_dir, type_path + f".target.processed{suffix}"), 'wb'))
        else:
            if os.path.exists(os.path.join('/workspace', type_path + ".source_ids.json")):
                print(f'loading {type_path}.source_ids.json from local to speed up!')
                self.source = json.load(open(os.path.join('/workspace', type_path + ".source_ids.json")))
            else:
                self.source = json.load(open(os.path.join(data_dir, type_path + ".source_ids.json")))
            self.target = encode_file(tokenizer, os.path.join(data_dir, type_path + ".target"), max_target_length)

        if self.generate_relevance:
            self.target = json.load(open(os.path.join(data_dir, type_path + ".relevance_labels.json")))

        self.all_answers = None
        if os.path.exists(os.path.join(data_dir, f"{type_path}.target.json")):
            self.all_answers = json.load(open(os.path.join(data_dir, f"{type_path}.target.json")))
            self.kw_labels_cache = {}

    def __len__(self):
        return len(self.source)

    def create_kw_labels(self, answers, target_ids):
        kw_labels = torch.zeros(target_ids.shape).type_as(target_ids)
        for a in answers:
            a_tokens = self.tokenizer.encode(a, add_special_tokens=False, return_tensors='pt')[0]
            a_len = a_tokens.shape[0]
            target_len = target_ids.shape[0]
            for idx in range(target_len - a_len):
                if torch.all(target_ids[idx: idx + a_len] == a_tokens):
                    kw_labels[idx: idx + a_len] = 1
        return kw_labels

    def select_psg(self, src):
        q_ids, ctx_ids_l = src
        if self.type_path == 'train':
            if random.random() <= 1:
                top_k = random.randrange(0, 11)
            else:
                top_k = 10
            selected_psg = list(range(top_k)) + random.choices(list(range(top_k, len(ctx_ids_l))), k=10 - top_k)
        else:
            selected_psg = range(10)
        source_ids = [self.tokenizer.bos_token_id] + q_ids + [self.tokenizer.eos_token_id]
        for idx in selected_psg:
            title_ids, text_ids = ctx_ids_l[idx]
            source_ids.extend(title_ids + [self.tokenizer.eos_token_id] + text_ids + [self.tokenizer.eos_token_id])
        source_ids = torch.LongTensor(source_ids[:1024])
        return source_ids

    def prepare_relevance_label(self, index, add_A=False):
        MAX_SRC_LENGTH = 256  # TODO change?
        q_ids, ctx_ids_l = self.source[index]
        source_ids = [self.tokenizer.bos_token_id] + q_ids + [self.tokenizer.eos_token_id]
        if add_A:
            As = self.all_answers[index]
            A_idx = random.randrange(0, len(As))
            a_tokens = self.tokenizer.encode(As[A_idx], add_special_tokens=False)
            source_ids.extend(a_tokens + [self.tokenizer.eos_token_id])

        ctx_idx = random.randrange(0, len(ctx_ids_l))
        title_ids, text_ids = ctx_ids_l[ctx_idx]
        source_ids.extend(title_ids + [self.tokenizer.eos_token_id] + text_ids + [self.tokenizer.eos_token_id])
        src_len = min(MAX_SRC_LENGTH, len(source_ids))
        pad_len = MAX_SRC_LENGTH - src_len
        source_ids = torch.LongTensor(source_ids[:MAX_SRC_LENGTH] + [self.tokenizer.pad_token_id] * pad_len)
        src_mask = torch.LongTensor([1] * src_len + [0] * pad_len)
        relevance_label = self.target[index][ctx_idx]
        if relevance_label:
            target_ids = self.tokenizer.encode('YES', return_tensors='pt')[0]
        else:
            target_ids = self.tokenizer.encode('NO', return_tensors='pt')[0]
        return source_ids, src_mask, target_ids

    def __getitem__(self, index):
        if self.concat_source:
            source_ids = self.source[index]["input_ids"].squeeze()
            target_ids = self.target[index]["input_ids"].squeeze()
            src_mask = self.source[index]["attention_mask"].squeeze()
            self.shuffle = False  # NB. whether shuffle input psg or not
            if self.shuffle and self.type_path == 'train':
                # no change to attention_mask since they are all 1s for top-10 psg
                idx_l = []
                last_idx = -1
                for ct, i in enumerate(torch.where(source_ids == 2)[0]):
                    # since title is split by 2 too
                    if ct % 2 == 0:
                        idx_l.append((last_idx + 1, i.item()))
                        last_idx = i.item()
                if last_idx != 1023:
                    idx_l.append((last_idx + 1, 1023))
                psg_idx_l = idx_l[1:]
                random.shuffle(psg_idx_l)
                idx_l = idx_l[:1] + psg_idx_l
                new_source_ids = []
                for start, end in idx_l:
                    new_source_ids.append(source_ids[start: end + 1])
                source_ids = torch.cat(new_source_ids)
        else:
            if self.generate_relevance:
                source_ids, src_mask, target_ids = self.prepare_relevance_label(index, add_A=True)
            else:
                target_ids = self.target[index]["input_ids"].squeeze()
                source_ids = self.select_psg(self.source[index]).type_as(target_ids)
                src_mask = torch.ones(1024).type_as(target_ids)

        # whether add kw_labels (mark answer spans) when generating psg [not used]
        kw_labels = None
        # if self.all_answers is not None:
        #     if index not in self.kw_labels_cache:
        #         answers = self.all_answers[index]
        #         self.kw_labels_cache[index] = self.create_kw_labels(answers, target_ids)
        #     kw_labels = self.kw_labels_cache[index]

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, 'kw_labels': kw_labels}

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id):
        y = trim_batch(batch["target_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(batch["source_ids"], pad_token_id, attention_mask=batch["source_mask"])
        return source_ids, source_mask, y

    def collate_fn(self, batch):
        input_ids = torch.stack([x["source_ids"] for x in batch])
        masks = torch.stack([x["source_mask"] for x in batch])
        target_ids = torch.stack([x["target_ids"] for x in batch])
        pad_token_id = self.tokenizer.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        if batch[0]['kw_labels'] is not None:
            kw_labels = torch.stack([x["kw_labels"] for x in batch])
            kw_labels = kw_labels[:, :y.shape[1]]
            return {"source_ids": source_ids, "source_mask": source_mask, "target_ids": y, 'kw_labels': kw_labels}
        return {"source_ids": source_ids, "source_mask": source_mask, "target_ids": y}


def freeze_params(model, except_para=None):
    if type(model) == dict:
        for name, par in model.items():
            if except_para is not None and except_para in name:
                par.requires_grad = True
            else:
                par.requires_grad = False
    else:
        for name, par in model.named_parameters():
            if except_para is not None and except_para in name:
                par.requires_grad = True
            else:
                par.requires_grad = False


def unfreeze_params(model, except_para=None):
    if type(model) == dict:
        for name, par in model.items():
            if except_para is not None and except_para in name:
                par.requires_grad = False
            else:
                par.requires_grad = True
    else:
        for name, par in model.named_parameters():
            if except_para is not None and except_para in name:
                par.requires_grad = False
            else:
                par.requires_grad = True

import copy

from tqdm import tqdm

from dpr.data.qa_validation import has_answer, has_answer_count
from dpr.utils.tokenizers import SimpleTokenizer

simple_tokenizer = SimpleTokenizer()


def calc_retrieval_acc(data, k=100, ctx_key="ctxs"):
    ct = 0
    for d in data:
        for ctx in d[ctx_key][:k]:
            if ctx["has_answer"]:
                ct += 1
                break
    return ct / len(data)


def rider_rerank(data, q2pred, n_ctxs=100, n_pred=5, key='ctxs_rerank_100psg', mode='normal'):
    """
    @param data: the retriever results in the same format of DPR
    @param q2pred: a dict with key=question and value=[a list of predicted answers of some reader]
    @param n_ctxs: num of passages to use
    @param n_pred: num of reader predictions to use
    @param key: the key under which reranked passages are stored
    @param mode: 'normal' is the method described in the paper,
    'interleave' tries to cover different answers by interleaving the matched psg,
    'density' consider #answers matched in each psg
    """
    assert mode in ['normal', 'interleave', 'density']

    for d in tqdm(data):
        d[key] = []
        ctx_l = [[] for _ in range(n_pred)]
        for ctx in d['ctxs'][:n_ctxs]:
            # if contains reader predictions, add to new ctx list
            pred = q2pred[d['question']]
            if mode == 'normal':
                has_answer_flag = has_answer(answers=set(pred[:n_pred]), text=ctx['title'] + ' ' + ctx['text'],
                                             tokenizer=simple_tokenizer, match_type='string')
                if has_answer_flag:
                    ctx_new = copy.deepcopy(ctx)
                    ctx_new['has_answer'] = has_answer(answers=d['answers'], text=ctx['title'] + ' ' + ctx['text'],
                                                       tokenizer=simple_tokenizer, match_type='string')
                    d[key].append(ctx_new)

            elif mode == 'interleave':
                # make the gen-reader see psg covering different answers
                for ctx_idx, p in enumerate(pred[:n_pred]):
                    has_answer_flag = has_answer(answers=[p], text=ctx['title'] + ' ' + ctx['text'],
                                                 tokenizer=simple_tokenizer, match_type='string')
                    if has_answer_flag:
                        ctx_new = copy.deepcopy(ctx)
                        ctx_new['has_answer'] = has_answer(answers=d['answers'], text=ctx['title'] + ' ' + ctx['text'],
                                                           tokenizer=simple_tokenizer, match_type='string')
                        ctx_l[ctx_idx].append(ctx_new)
                        break

            elif mode == 'density':
                num_answers = has_answer_count(answers=set(pred[:n_pred]), text=ctx['title'] + ' ' + ctx['text'],
                                               tokenizer=simple_tokenizer, match_type='string')
                if num_answers > 0:
                    ctx_new = copy.deepcopy(ctx)
                    ctx_new['has_answer'] = has_answer(answers=d['answers'], text=ctx['title'] + ' ' + ctx['text'],
                                                       tokenizer=simple_tokenizer, match_type='string')
                    d[key].append((ctx_new, num_answers))

        if mode == 'density':
            # sort by num_answers
            d[key] = sorted(d[key], key=lambda x: x[1], reverse=True)
            d[key] = [i[0] for i in d[key]]

        if mode == 'interleave':
            # add psg covering each answer one by one
            idx_l = [0 for _ in range(n_pred)]
            finished = False
            while not finished:
                finished = True
                for i in range(len(idx_l)):
                    idx = idx_l[i]
                    if idx < len(ctx_l[i]):
                        finished = False
                        d[key].append(ctx_l[i][idx])
                        idx_l[i] += 1

        id_set = set([ctx['id'] for ctx in d[key]])
        for ctx in d['ctxs'][:n_ctxs]:
            if ctx['id'] not in id_set:
                d[key].append(copy.deepcopy(ctx))

    print('\t\t old   rerank')
    for k in [1, 5, 10, 20, 100]:
        acc = calc_retrieval_acc(data, k=k)
        acc_rerank = calc_retrieval_acc(data, k=k, ctx_key=key)
        print(f'top-{k} acc:\t {acc:.3f} {acc_rerank:.3f}')


def write_rerank_psg(data, fname, ctx_key="ctxs_rerank_100psg"):
    """
    write reranked passages to file as the input of the generative reader
    """
    with open(fname, "w") as o:
        for d in data:
            s = d["question"] + " </s> "
            for idx in range(10):
                s += (
                        d[ctx_key][idx]["title"]
                        + " </s> "
                        + d[ctx_key][idx]["text"]
                        + " </s> "
                )
            o.write(s + "\n")

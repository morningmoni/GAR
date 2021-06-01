This repo provides the code of the following papers:

(**GAR**) ["Generation-Augmented Retrieval for Open-domain Question Answering"](https://arxiv.org/abs/2009.08553), ACL 2021

(**RIDER**) ["Reader-Guided Passage Reranking for Open-Domain Question Answering"](https://arxiv.org/abs/2101.00294), Findings of ACL 2021.



GAR augments a question with relevant contexts generated by seq2seq learning, with the question as input and target outputs such as the answer, the sentence where the answer belongs to, and the title of a passage that contains the answer. With the generated contexts appended to the original questions, GAR achieves state-of-the-art OpenQA performance with a simple BM25 retriever.

RIDER is a simple and effective passage reranker, which reranks retrieved passages by reader predictions without any training. RIDER achieves 10~20 gains in top-1 retrieval accuracy, 1~4 gains in Exact Match (EM), and even outperforms supervised transformer-based rerankers.



## Code

### Generation

The codebase of seq2seq models is based on (old) [huggingface](https://github.com/huggingface)/[transformers](https://github.com/huggingface/transformers) (version==2.11.0) examples. 

**See  `train_gen.yml` for the package requirements and example commands to run the models.** 

`train_generator.py`: training of seq2seq models.

`conf.py`: configurations for `train_generator.py`.  There are some default parameters but it might be easier to set e.g., `--data_dir` and `--output_dir` directly.

`test_generator.py`: test of seq2seq models (if not already done in `train_generator.py`).



### Retrieval

We use [pyserini](https://github.com/castorini/pyserini) for BM25 retrieval. Please refer to its [document](https://github.com/castorini/pyserini/#how-do-i-index-and-search-my-own-documents) for indexing and searching wiki passages (wiki passages can be downloaded [here](https://github.com/facebookresearch/DPR#resources--data-formats)). Alternatively, you may take a look at its [effort to reproduce DPR results](https://github.com/castorini/pyserini/blob/master/docs/experiments-dpr.md), which gives more detailed instructions and incorporates the passage-level span voting in GAR.



### Reranking

Please see the instructions in `rider/rider.py`.



### Reading

We experiment with one extractive reader and one generative reader. 

For the extractive reader, we take the one used by dense passage retrieval. Please refer to [DPR](https://github.com/facebookresearch/DPR) for more details. 

For the generative reader, we reuse the codebase in the generation stage above, with [question; top-retrieved passages] as the source input and one ground-truth answer as the target output. Example script is provided in `train_gen.yml`.



## Data

Please refer to [DPR](https://github.com/facebookresearch/DPR#resources--data-formats) for dataset downloading.

For seq2seq learning, use {train/val/test}.source as the input and {train/val/test}.target as the output, where each line is one example. 

In the same folder, save the list of ground-truth answers with name {val/test}.target.json if you want to evaluate EM during training.



## Cite

Please use the following bibtex to cite our papers. 

```
@article{mao2020generation,
  title={Generation-augmented retrieval for open-domain question answering},
  author={Mao, Yuning and He, Pengcheng and Liu, Xiaodong and Shen, Yelong and Gao, Jianfeng and Han, Jiawei and Chen, Weizhu},
  journal={arXiv preprint arXiv:2009.08553},
  year={2020}
}

@article{mao2021reader,
  title={Reader-Guided Passage Reranking for Open-Domain Question Answering},
  author={Mao, Yuning and He, Pengcheng and Liu, Xiaodong and Shen, Yelong and Gao, Jianfeng and Han, Jiawei and Chen, Weizhu},
  journal={arXiv preprint arXiv:2101.00294}
}

```




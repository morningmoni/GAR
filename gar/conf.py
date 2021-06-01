from pathlib import Path
import os


def setup_task():
    # set target
    dataset = 'nq'
    mode = 'multi-inputs'
    if os.getenv('GEN_DATASET') is not None:
        dataset = os.getenv('GEN_DATASET')
    if os.getenv('GEN_TARGET') is not None:
        mode = os.getenv('GEN_TARGET')

    # set path
    data_path = os.getenv('PT_DATA_DIR')
    output_path = os.getenv('PT_OUTPUT_DIR')
    if data_path is None:
        mount_path = '/mount/data'
        if os.path.exists(mount_path):
            print('setting mount path')
            data_path = mount_path
            output_path = mount_path + '/output'
        else:
            print('setting local path')
            data_path = '/workspace/GAR/'
            output_path = '/workspace/GAR/output'
    output_path = Path(output_path)

    # default target if not set
    if dataset is None:
        print('dataset is not set, setting to [nq] by default!')
        mode = 'nq'
    else:
        print(f'dataset=[{dataset}]')
    if mode is None:
        print('mode is not set, setting to [answer] by default!')
        mode = 'answer'
    else:
        print(f'mode=[{mode}]')
    remark = f'{dataset}_{mode}'

    # per target parameters
    bs = 128
    bs_eval = 256
    if 'multi-inputs' in mode:
        bs = 8
        bs_eval = 16
    if dataset == 'nq':
        MAX_SRC_LENGTH = 20  # 99%
        if mode == 'title':
            ckpt_name = Path(data_path) / "checkpointepoch=XXX.ckpt"
            data_path = Path(data_path) / 'nq-title/'
            # MAX_TGT_LENGTH = 32  # < 50%
            MAX_TGT_LENGTH = 64  # 80% ~ 95%
        elif mode == 'answer':
            ckpt_name = Path(data_path) / "checkpointepoch=XXX.ckpt"
            data_path = Path(data_path) / 'nq-answer/'
            # MAX_TGT_LENGTH = 10  # 75%
            MAX_TGT_LENGTH = 40  # 99%
        elif mode == 'sentence':
            ckpt_name = Path(data_path) / "checkpointepoch=XXX.ckpt"
            data_path = Path(data_path) / 'nq-sentence/'
            MAX_TGT_LENGTH = 64  # 80% ~ 85%
        elif mode == 'multi-inputs':
            MAX_SRC_LENGTH = 1024
            MAX_TGT_LENGTH = 10  # 99% for answers[0]
            ckpt_name = Path(data_path) / "checkpointepoch=XXX.ckpt"
            data_path = Path(data_path) / 'nq-multi-inputs/'
        else:
            raise NotImplementedError
    elif dataset == 'trivia':
        MAX_SRC_LENGTH = 50  # 95%~99%
        if mode == 'title':
            ckpt_name = Path(data_path) / "checkpointepoch=XXX.ckpt"
            data_path = Path(data_path) / 'trivia-title/'
            MAX_TGT_LENGTH = 64  # 80%
        elif mode == 'answer':
            ckpt_name = Path(data_path) / "checkpointepoch=XXX.ckpt"
            data_path = Path(data_path) / 'trivia-answer'
            MAX_TGT_LENGTH = 16  # 99%
        elif mode == 'sentence':
            ckpt_name = Path(data_path) / "checkpointepoch=XXX.ckpt"
            data_path = Path(data_path) / 'trivia-sentence/'
            MAX_TGT_LENGTH = 64  # 80% ~ 95%
        elif mode == 'multi-inputs':
            MAX_SRC_LENGTH = 1024
            MAX_TGT_LENGTH = 10  # 90%~95% for answers[0], 95% for value
            ckpt_name = Path(data_path) / "checkpointepoch=XXX.ckpt"
            data_path = Path(data_path) / 'trivia-multi-inputs/'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return mode, output_path, data_path, MAX_SRC_LENGTH, MAX_TGT_LENGTH, remark, bs, bs_eval, ckpt_name


def add_generic_args(parser, root_dir):
    mode, output_path, data_path, MAX_SRC_LENGTH, MAX_TGT_LENGTH, remark, bs, bs_eval, ckpt_name = setup_task()

    parser.add_argument(
        "--fp16",
        action='store_true',
        default=True,
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument("--n_tpu_cores", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--do_train", default=True, action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", default=True, action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--ckpt_metric", type=str, default='val-acc')
    parser.add_argument("--ckpt_mode", type=str, default='max')
    parser.add_argument("--save_top_k", type=int, default=1)

    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--n_gpu", type=int, default=1)
    # parser.add_argument("--nproc_per_node", type=int, default=2)
    parser.add_argument("--min_gpu_memory", type=int, default=13000)

    parser.add_argument("--num_train_epochs", default=100, type=int)
    parser.add_argument("--train_batch_size", default=bs, type=int)
    parser.add_argument("--eval_batch_size", default=bs_eval, type=int)

    parser.add_argument("--remark", default=remark, type=str)
    parser.add_argument("--data_dir", default=data_path, type=str)
    parser.add_argument("--output_dir", default=output_path / remark, type=str)
    parser.add_argument("--ckpt_name", default=ckpt_name, type=str)
    parser.add_argument("--load_ckpt_name", default=None, type=str)
    parser.add_argument("--mode_real", type=str, default='seq2seq')
    # NB. change to 1e-5 / 5e-6 for multi-inputs
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_source_length", default=MAX_SRC_LENGTH, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=MAX_TGT_LENGTH, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--logger_name", default='default', type=str, choices=["default", "wandb"])


def add_model_specific_args(parser, root_dir):
    parser.add_argument(
        "--model_name_or_path",
        default='facebook/bart-large',
        # default='t5-large',
        # default='t5-3b',
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--encoder_layerdrop",
        type=float,
        help="Encoder layer dropout probability (Optional). Goes into model.config",
    )
    parser.add_argument(
        "--decoder_layerdrop",
        type=float,
        help="Decoder layer dropout probability (Optional). Goes into model.config",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        help="Dropout probability (Optional). Goes into model.config",
    )
    parser.add_argument(
        "--attention_dropout",
        type=float,
        help="Attention dropout probability (Optional). Goes into model.config",
    )
    parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
    parser.add_argument("--val_check_interval", type=float, default=1.0, required=False)
    parser.add_argument("--limit_val_batches", type=float, default=1.0, required=False)
    parser.add_argument("--label_decay", type=float, default=1, required=False, help="reweight token losses")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1, required=False)
    parser.add_argument("--freeze_embeds", action="store_true")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--freeze_decoder", action="store_true")

import argparse


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # Model
    #model_arg = parser.add_argument_group('Model')
    parser.add_argument('--n_head',
                           type=int, default=8,
                           help='GPT number of heads')
    parser.add_argument('--fold',
                           type=int, default=0,
                           help='number of folds for fine tuning')
    parser.add_argument('--n_layer',
                           type=int, default=12,
                           help='GPT number of layers')
    parser.add_argument('--d_dropout',
                           type=float, default=0.1,
                           help='Decoder layers dropout')
    parser.add_argument('--n_embd',
                           type=int, default=768,
                           help='Latent vector dimensionality')
    parser.add_argument('--fc_h',
                           type=int, default=512,
                           help='Fully connected hidden dimensionality')


    # Train
    #train_arg = parser.add_argument_group('Train')
    parser.add_argument('--n_batch',
                           type=int, default=512,
                           help='Batch size')
    parser.add_argument('--from_scratch',
                           action='store_true', default=False,
                           help='train on qm9 from scratch')
    parser.add_argument('--checkpoint_every',
                           type=int, default=1000,
                           help='save checkpoint every x iterations')
    parser.add_argument('--lr_start',
                           type=float, default=3 * 1e-4,
                           help='Initial lr value')
    parser.add_argument('--lr_multiplier',
                           type=int, default=1,
                           help='lr weight multiplier')
    parser.add_argument('--n_jobs',
                           type=int, default=1,
                           help='Number of threads')
    parser.add_argument('--device',
                        type=str, default='cuda',
                        help='Device to run: "cpu" or "cuda:<device number>"')
    parser.add_argument('--seed',
                        type=int, default=12345,
                        help='Seed')

    parser.add_argument('--seed_path',
                           type=str, default="",
                           help='path to  trainer file to continue training')

    parser.add_argument('--num_feats',
                            type=int, required=False, default=32,
                            help='number of random reatures for FAVOR+')
    parser.add_argument('--max_epochs',
                            type=int, required=False, default=1,
                            help='max number of epochs')

    # debug() FINE TUNEING
    # parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--mode',
                           type=str, default='avg',
                           help='type of pooling to use')
    parser.add_argument("--train_dataset_length", type=int, default=None, required=False)
    parser.add_argument("--eval_dataset_length", type=int, default=None, required=False)
    parser.add_argument("--desc_skip_connection", type=bool, default=False, required=False)
    parser.add_argument("--num_workers", type=int, default=8, required=False)
    parser.add_argument("--dropout", type=float, default=0.1, required=False)
    parser.add_argument("--dims", type=int, nargs="*", default="[]", required=False)
    parser.add_argument(
        "--smiles_embedding",
        type=str,
        default="/dccstor/medscan7/smallmolecule/runs/ba-predictor/small-data/embeddings/protein/ba_embeddings_tanh_512_2986138_2.pt",
    )
    # parser.add_argument("--train_pct", type=str, required=False, default="95")
    parser.add_argument("--aug", type=int, required=False)
    parser.add_argument("--num_classes", type=int, required=False)
    parser.add_argument("--dataset_name", type=str, required=False, default="sol")
    parser.add_argument("--measure_name", type=str, required=False, default="measure")
    parser.add_argument("--checkpoints_folder", type=str, required=True)
    parser.add_argument("--checkpoint_root", type=str, required=False)

    parser.add_argument(
        "--data_root",
        type=str,
        required=False,
        default="/dccstor/medscan7/smallmolecule/runs/ba-predictor/small-data/affinity",
    )
    parser.add_argument("--batch_size", type=int, default=64)

    return parser
def parse_args():
    parser = get_parser()
    args = parser.parse_args()
    return args


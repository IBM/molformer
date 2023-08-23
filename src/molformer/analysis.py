from src.molformer.utils import get_argparse_defaults, read_config
from src.molformer.model.args import get_parser
from src.molformer.tokenizer import MolTranBertTokenizer
from src.molformer.model.base_bert import get_bert
from src.molformer.model.args import parse_args as ARGS
from argparse import AttributeDict

# build|pipeline

### defaults
DEFAULTS = AttributeDict(get_argparse_defaults(ARGS))
DEFAULTS.seed_path = CHECKPOINT_PATH

### read|config|file
CONFIG_FILE_PATH = ''
CONFIG = read_config(CONFIG_FILE_PATH)

#?  config.canonical = False
#?  config.mask = False

### tokenizer
VOCAB_PATH = 'data/bert_vocab.txt'
TOKENIZER = MolTranBertTokenizer(VOCAB_PATH)

### import|model
BERT = get_bert(DEFAULTS, TOKENIZER)




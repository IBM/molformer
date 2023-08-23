from molformer.utils import get_argparse_defaults, read_config
from molformer.model.args import get_parser
from molformer.tokenizers import MolTranBertTokenizer
from molformer.model.modeling_bert import get_bert
from molformer.model.args import parse_args as ARGS
from argparse import AttributeDict
from molformer.data import bert_vocab

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
VOCAB_PATH = ''
TOKENIZER = MolTranBertTokenizer(VOCAB_PATH)

### import|model
BERT = get_bert(DEFAULTS, TOKENIZER)




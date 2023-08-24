from argparse import Namespace
import yaml
from rdkit import Chem


def read_config(hparams):
    with open(hparams) as f:
        config = Namespace(**yaml.safe_load(f))

    return config


def get_argparse_defaults(parser):
    defaults = {}
    for action in parser._actions:
        if not action.required and action.dest != "help":
            defaults[action.dest] = action.default
    return defaults


def normalize_smiles(smi, canonical, isomeric):
    normalized = Chem.MolToSmiles(
        Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric
    )
    return normalized

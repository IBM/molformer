from argparse import Namespace
import yaml


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

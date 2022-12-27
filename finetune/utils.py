import debugpy
import socket
import glob
import pandas as pd
from typing import List
from rdkit import Chem


def getipaddress():
    return socket.gethostbyname(socket.getfqdn())


def debug():
    print("Waiting for debugger to connect")
    if (
        socket.getfqdn().startswith("dcc")
        or socket.getfqdn().startswith("mol")
        or socket.getfqdn().startswith("ccc")
    ):
        debugpy.listen(address=(getipaddress(), 3000))
        debugpy.wait_for_client()
    debugpy.breakpoint()


class ListDataset:
    def __init__(self, seqs):
        self.seqs = seqs

    def __getitem__(self, index):
        return self.seqs[index]

    def __len__(self):
        return len(self.seqs)


def transform_single_embedding_to_multiple(smiles_z_map):
    """Transforms an embedding map of the format smi->embedding to
    smi-> {"canonical_embeddings":embedding}. This function exists
    as a compatibility layer

    Args:
        smiles_z_map ([type]): [description]
    """
    retval = dict()
    for key in smiles_z_map:
        retval[key] = {"canonical_embeddings": smiles_z_map[key]}
    return retval


def normalize_smiles(smi, canonical, isomeric):
    normalized = Chem.MolToSmiles(
        Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric
    )
    return normalized


def get_all_proteins(affinity_dir: str):
    files = glob.glob(affinity_dir + "/*.csv")
    all_proteins = []
    print(files)
    for file in files:
        df = pd.read_csv(file)
        all_proteins.extend(df["protein"].tolist())
    return set(all_proteins)


def append_to_file(filename, line):
    with open(filename, "a") as f:
        f.write(line + "\n")


def write_to_file(filename, line):
    with open(filename, "w") as f:
        f.write(line + "\n")

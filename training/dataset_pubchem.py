import torch
from rdkit import Chem
import codecs
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import time

class SS:
    bos = '<bos>'
    eos = '<eos>'
    pad = '<pad>'
    unk = '<unk>'

class DatasetPubchem(torch.utils.data.IterableDataset):
#class DatasetPubchem(torch.utils.data.Dataset):
    
    def __init__(self, train_load=None, vocab=None,  randomize_smiles=False):
        """PubChem Dataset


        Keyword Arguments:
            database_file {[type]} -- [description]
            randomize_smiles {bool} -- Randomize the smiles each epoch
        """
        #regex vocab used by Molecular Transformer
        import re
        pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(pattern)

        if train_load is None:
            self.database_file = '/dccstor/trustedgen/data/pubchem/CID-SMILES' 
        else:
            self.database_file = train_load
        self.len = len(open(self.database_file, 'rb').readlines())
        #self.smiles = []
        #with codecs.open(self.database_file) as f:
        #    for line in f:
        #        self.smiles.append(''.join(self.regex.findall(line.split()[-1])))
            
        #self.smiles = pd.read_csv(self.database_file)[smiles_header_name].tolist()
        self.randomize_smiles = randomize_smiles
        if vocab != None:
            self.vocab = vocab 
        else:
            self.vocab = {}
            vocab = torch.load('./atom_vocab.pth')
            self.ss = SS
            self.c2i = {c: i for i, c in vocab.items()}
            self.i2c = {i: c for i, c in vocab.items()}

    def process(self,text):
            #return np.asarray(self.string2ids(self.smi_tokenizer(smiles), add_bos=True, add_eos=True))
        print(text)
        mol = ''.join(self.regex.findall(text.split()[-1]))
        return mol
        #return np.asarray(self.string2ids(self.smi_tokenizer(mol), add_bos=True, add_eos=True))
    def line_mapper(self, line):
        text = self.process(line)
        return text
    def __len__(self):
        #pass
        return self.len
    #    return len(self.smiles)
    def __iter__(self):
        data_file = open(self.database_file)
        data_map = map(self.line_mapper, data_file)
        return data_map
    
    def get_vocab(self):
        return self.c2i
        #return {'bos':self.ss.bos, 'eos':self.ss.eos, 'pad':self.ss.pad,
                #'unk':self.ss.unk, 'c2i':self.c2i, 'i2c':self.i2c}
        
    def smi_tokenizer(self, smi):
        """
        Tokenize a SMILES molecule or reaction
        """
        tokens = self.regex.findall(smi)
        #tokens = [token for token in self.regex.findall(smi)]
        assert smi == ''.join(tokens)
        return tokens

    #def __getitem__(self, index):
    #    smiles = self.smiles[index]
    #    #print(smiles)
    #    #print(self.vocab.string2ids(smiles, add_bos=True, add_eos=True))
    #    #print(np.asarray(self.vocab.string2ids(smiles, add_bos=True, add_eos=True)))
    #    
    #    if self.randomize_smiles: 
    #        smiles = self.randomize_smiles(smiles, self.isomeric_smiles)
    #    if self.is_measure_available:
    #        return smiles, self.measure[index]
    #    else:
    #        return np.asarray(self.string2ids(self.smi_tokenizer(smiles), add_bos=True, add_eos=True))


    def create_collate_fn(self, pad):
        def collate(batch):
            '''
            Padds batch of variable length
            note: it converts things ToTensor manually here since the ToTensor transform
            assume it takes in images rather than arbitrary tensors.
                            '''
            ## get sequence lengths
            lengths = torch.tensor([ t.shape[0] for t in batch if t.shape[0] > 4 and t.shape[0] < 42 ]])
            batch_tmp = [ torch.from_numpy(t) for t in batch if t.shape[0] > 4 and t.shape[0] < 42 ]
            batch_tmp = torch.nn.utils.rnn.pad_sequence(batch_tmp, batch_first=True, padding_value=pad)
            target = [ torch.from_numpy(t[1:]) for t in batch if t.shape[0] > 4 and t.shape[0] < 42 ]
            target = torch.nn.utils.rnn.pad_sequence(target, batch_first=True, padding_value=pad)
            target = torch.cat((target, torch.full((target.size(0), 1),pad, dtype=int)), dim=-1)
            lengths_mid = None
            batch_tmp_mid = None
            target_mid = None 
            assert(False)
            if len(batch_tmp) < len(batch):
                lengths_mid = torch.tensor([ t.shape[0] for t in batch if t.shape[0] >= 42 and t.shape[0] <= 100 ]])
                batch_tmp_mid = [ torch.from_numpy(t) for t in batch if t.shape[0] >= 42 and t.shape[0] <= 100 ]
                batch_tmp_mid = torch.nn.utils.rnn.pad_sequence(batch_tmp_mid, batch_first=True, padding_value=pad)
                target_mid = [ torch.from_numpy(t[1:]) for t in batch if t.shape[0] >= 42  and t.shape[0] <= 100 ]
                target_mid= torch.nn.utils.rnn.pad_sequence(target, batch_first=True, padding_value=pad)
                target_mid= torch.cat((target, torch.full((target.size(0), 1),pad, dtype=int)), dim=-1)
            ## compute mask
            #mask = (batch != pad)
            lengths_long = None
            batch_tmp_long = None
            target_long = None 

            print(len(batch_tmp) < len(batch))
            if len(batch_tmp) < len(batch):
                lengths_long = torch.tensor([ t.shape[0] for t in batch if t.shape[0] >= 101 and t.shape[0] <= 200 ]])
                batch_tmp_long = [ torch.from_numpy(t) for t in batch if t.shape[0] >= 101 and t.shape[0] <= 200 ]
                batch_tmp_long = torch.nn.utils.rnn.pad_sequence(batch_tmp_long, batch_first=True, padding_value=pad)
                target_long = [ torch.from_numpy(t[1:]) for t in batch if t.shape[0] >= 101  and t.shape[0] <= 200 ]
                target_long= torch.nn.utils.rnn.pad_sequence(target, batch_first=True, padding_value=pad)
                target_long= torch.cat((target, torch.full((target.size(0), 1),pad, dtype=int)), dim=-1)
            ## compute mask
            return batch_tmp, lengths, target, batch_tmp_mid, lengths_mid, target_mid, batch_tmp_long, lengths_long, target_long 
        return collate
    @property
    def bos(self):
        return self.c2i[self.ss.bos]

    @property
    def eos(self):
        return self.c2i[self.ss.eos]

    @property
    def pad(self):
        return self.c2i[self.ss.pad]

    @property
    def unk(self):
        return self.c2i[self.ss.unk]

    def char2id(self, char):
        if char not in self.c2i:
            return self.unk

        return self.c2i[char]

    def id2char(self, id):
        if id not in self.i2c:
            return self.unk

        return self.i2c[id]

    def string2ids(self, string, add_bos=False, add_eos=False):
        ids = [self.char2id(c) for c in string]

        if add_bos:
            ids = [self.bos] + ids
        if add_eos:
            ids = ids + [self.eos]

        return ids

    def ids2string(self, ids, rem_bos=True, rem_eos=True):
        if len(ids) == 0:
            return ''
        if rem_bos and ids[0] == self.bos:
            ids = ids[1:]
        if rem_eos and ids[-1] == self.eos:
            ids = ids[:(ids != self.eos).sum()]
            #ids = ids[:-1]

        string = ''.join([self.id2char(id) for id in ids])

        return string
    def read_smiles_csv(path, fields=None, normalize=True):
        df = pd.read_csv(path)
        fields=['SMILES']
        data = [df[field].to_list() for field in fields]
        return data
        #return list(zip(*data))
    def randomize_smiles(self, smiles, isomeric_smiles):
        """Perform a randomization of a SMILES string
        must be RDKit sanitizable"""
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None # Invalid SMILES
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m,ans)
        return Chem.MolToSmiles(nm, canonical=False, isomericSmiles=isomeric_smiles)
if __name__ == '__main__':
    
    t0 = time.time()
    print('start time is {}'.format(t0)) 
    dataset = DatasetPubchem()
    t1 = time.time()
    print('total time is {}'.format(t1-t0)) 
    print('len(dataset)')
    print(len(dataset))

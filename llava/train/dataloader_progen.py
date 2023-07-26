import torch
from tqdm import tqdm
from glob import glob
import pickle
import sys
from base_dataset import BaseDataset

class CATH_ESM_Dataset(BaseDataset):
    def __init__(self, ids, seqs, coords, embeddings) -> None:
        super().__init__()
        self.ids = ids
        self.seqs = seqs
        self.coords = coords
        self.embeddings = embeddings

        assert (len(self.embeddings) - len(self.seqs) + len(self.ids) - len(self.embeddings)) == 0, \
            'dataset err.'
        print(f'build dataset with {len(self.seqs)} proteins.')

    @classmethod
    def from_pdb(cls, pdb_path, prefix=None):
        data = None

    @classmethod
    def from_folder(cls, path, max_len=500):
        ids = []
        seqs = []
        coords = []
        embeddings = []
        cls.max_len = max_len
        print(f'load dataset from {path}.')

        data_path = sorted(glob(path+'/*.pkl'))

        for index, data in tqdm(enumerate(data_path)):
            with open(data, 'rb') as f:
                data = pickle.load(f)
            if len(data['seq']) >= max_len:
                continue
            ids.append(data['id']) if data['id'] is not None else index
            seqs.append(data['seq']) if data['seq'] is not None else print(f'seq err at {index}')
            # coords.append(data['coord']) if data['coord'] is not None else coords.append(torch.tensor([0]))
            embeddings.append(data['emb'].detach().cpu())
            # if index > 10:
            #     break

        return cls(ids, seqs, coords, embeddings)

    
    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        return {
            'embedding': self.embeddings[index],
            'sequence': self.seqs[index],
            # 'coords': self.coords[index]
        }
    
    def collator(self, samples):
        batch_size = len(samples)
        
        embedding_size = samples[0]['embedding'].shape[-1]
        # initialize 
        
        embeddings = torch.zeros(
            (
                batch_size,
                self.max_len,
                embedding_size
            )
        )
        attn_mask = torch.zeros(
            (
                batch_size,
                self.max_len
            )
        )
        seqs = []
        for idx, sample in enumerate(samples):
            embedding = sample['embedding']
            sequence = sample['sequence']
            seq_len = len(embedding)
            attn_mask[idx, :seq_len] = 1
            embeddings[idx, :seq_len] = embedding
            seqs.append(sequence)
        
        return {
            'emb': embeddings,
            'seq': seqs,
            'mask': attn_mask
        }
from typing import *
from pathlib import Path
import pandas as pd
from datasets import Dataset

try:
    from .vocab import Vocab
except:
    from vocab import Vocab

class TopicReorder():
    def __init__(self, vocab_dir: Union[str, Path], data_dir: Union[str, Path]) -> None:
        self.origin_file = Path(data_dir) / "subgroups.csv"
        df =  pd.read_csv(self.origin_file)
        self.origin_decoder = {
            id: name for id, name in zip(df['subgroup_id'], df['subgroup_name'])
        }

        self.vocab_file = Path(vocab_dir) / "subgroup.json"
        self.encoder = Vocab()
        self.encoder.load( self.vocab_file )


    def decode_origin_index(self, origin_idxs: List[str], with_bos_eos: bool) -> List[ List[str] ]:
        decoded = []
        for subgroups in origin_idxs:
            if subgroups:
                S = [self.origin_decoder[int(x)] for x in subgroups.split(' ') ]
                decoded.append( S )
            elif with_bos_eos:
                decoded.append( [] )
            else:
                decoded.append( [ Vocab.UNK ] )

        return decoded


    def encode_subgroups(self, decoded_subgroups: List[List[str]], with_bos_eos: bool) -> List[ List[int] ]:
        encoded = []
        for subgroups in decoded_subgroups:
            # after decode, the subgroup should be List[str] without None. ex: [] or ["xxx", "aaa"]...
            if with_bos_eos:
                subgroups = [self.encoder.BOS] + subgroups + [self.encoder.EOS]

            S = self.encoder.encode( subgroups )
            encoded.append( S )

        return encoded

    def process_dataset(self, dataset: Dataset, with_bos_eos: bool) -> Dataset:
        decoded = self.decode_origin_index(dataset['subgroup'], with_bos_eos)
        encoded = self.encode_subgroups(decoded, with_bos_eos)
        return dataset.remove_columns(['subgroup']).add_column('subgroup', encoded)


if __name__ == "__main__":
    TR = TopicReorder("../../cache/vocab", "../../data")
    D = Dataset.from_csv("../../data/train_group.csv")
    D = D.select(range(50, 60))

    for flag in [False, True]:
        decoded = TR.decode_origin_index(D['subgroup'], flag)
        encoded = TR.encode_subgroups(decoded, flag)
        print()
        for dec, enc in zip(decoded, encoded):
            print(f"{dec} -> {enc}")
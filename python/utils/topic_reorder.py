from interface import Preprocessor
from typing import *
from pathlib import Path
from vocab import Vocab
import pandas as pd
from datasets import Dataset


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


    def decode_origin_index(self, origin_idxs: List[str]) -> List[ List[str] ]:
        decoded = []
        for subgroups in origin_idxs:
            if subgroups:
                S = [self.origin_decoder[int(x)] for x in subgroups.split(' ') ]
                decoded.append( S )
            else:
                decoded.append( [ Vocab.UNK ] )
        return decoded


    def encode_subgroups(self, decoded_subgroups: List[List[str]]) -> List[ List[int] ]:
        encoded = []
        for subgroups in decoded_subgroups:
            if subgroups:
                S = self.encoder.encode( subgroups )
                encoded.append( S )
            else:
                encoded.append(subgroups)
        return encoded

    def process_dataset(self, dataset: Dataset) -> Dataset:
        decoded = self.decode_origin_index(dataset['subgroup'])
        encoded = self.encode_subgroups(decoded)
        return dataset.remove_columns(['subgroup']).add_column('subgroup', encoded)


if __name__ == "__main__":
    TR = TopicReorder("../../cache/vocab", "../../data")
    D = Dataset.from_csv("../../data/train_group.csv")

    decoded = TR.decode_origin_index(D['subgroup'])
    encoded = TR.encode_subgroups(decoded)

    print(D['subgroup'][:5])
    print(decoded[:5])
    print(encoded[:5])

    D = TR.process_dataset(D)
    print(D[:5])

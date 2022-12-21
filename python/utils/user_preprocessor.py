from typing import *
from pathlib import Path
import json
import datasets
from datasets import Dataset
from transformers import (
    PreTrainedTokenizer,
    AutoTokenizer,
)
try:
    from .interface import Preprocessor
    from .vocab import Vocab
except:
    from interface import Preprocessor
    from vocab import Vocab

class UserPreprocessor( Preprocessor ):
    """
        - user_id
        - gender: ['male', 'female', 'others']
        - occupation_titles: "藝文設計,服務業"
        - interests: "藝術_角色設計,藝術_電腦繪圖,藝術_繪畫與插畫,設計_平面設計"
        - recreation_names: "手作,手寫字,插畫,水彩,貓派,速寫,閱讀,電影,電玩,電腦繪圖"
    """
    
    def __init__(self, column_names = ['user_id', 'gender', 'occupation_titles', 'interests', 'recreation_names']):
        # Column name should follow order: uid, gender, title, interests, recreations
        self.column_names = column_names
        self.__encode_func__ = [self.encode_user_id, self.encode_gender, self.encode_titles, self.encode_interests, self.encode_recreations]
        self.__none_value__ = [
            Vocab.UNK,
            Vocab.UNK,
            Vocab.UNK,
            f"{Vocab.UNK}_{Vocab.UNK}",
            Vocab.UNK,
        ]

    @staticmethod
    def interest_generator(interests: str):
        for interest in interests.split(','):
            group, subgroup = interest.split('_')
            yield (group, subgroup)

    def fill_none_as_unk(self,  batch: Dict[str, List[str]]) -> Dict[str, List[str]]:
        for column, value in zip(self.column_names, self.__none_value__):
            batch[column] = [
                data if data != None else value
                for data in batch[column]
            ]
        return batch

    def append_bos_eos(self, batch: Dict[str, List[str]]) -> Dict[str, List[str]]:
        for column, value in zip(self.column_names, self.__none_value__):
            if column in ['user_id', 'gender']:
                res = [ data if data != None else value for data in batch[column] ]
            elif column in ['interests']:
                res = [
                    ",".join( [f"{Vocab.BOS}_{Vocab.BOS}", data, f"{Vocab.EOS}_{Vocab.EOS}"] )
                    if data != None else value
                    for data in batch[column]
                ]
            else:
                res = [
                    ",".join( [Vocab.BOS, data, Vocab.EOS] )
                    if data != None else value
                    for data in batch[column]
                ]

            batch[column] = res
        return batch

    def encode(self, user_profile: Dict[str, List[str]]) -> Dict[str, List[ Union[int, List[int]] ]]:
        res = {
            key: func( user_profile[key] ) for key, func in zip(self.column_names, self.__encode_func__)
        }
        interest_col = self.column_names[-2]
        res['groups'] = res[ interest_col ][0]
        res['subgroups'] = res[ interest_col ][1]
        del res[ interest_col ]
        return res

    def encode_user_id(self, user_id: List[str]):
        raise NotImplementedError
    def encode_gender(self, gender: List[str]):
        raise NotImplementedError
    def encode_titles(self, titles: List[str]):
        raise NotImplementedError
    def encode_interests(self, interests: List[str]):
        raise NotImplementedError
    def encode_recreations(self, recreations: List[str]):
        raise NotImplementedError



class BasicUserPreprocessor( UserPreprocessor ):
    """
        mapping all attribute into idx
    """
    def __init__(self, vocab_dir: Union[str, Path], column_names):
        super().__init__(column_names)

        self.vocab_dir = vocab_dir
        if type(self.vocab_dir) == str:
            self.vocab_dir = Path(self.vocab_dir)

        self.files = [
            'user.json',
            'gender.json',
            'title.json',
            'group.json',
            'subgroup.json',
            'recreation.json',
        ]

        self.encoder: Dict[str, Vocab] = {}
        
        for file in self.files:
            file_path: Path = self.vocab_dir / file
            assert file_path.exists(), f"{file_path} doesn't exist"
            vocab = Vocab()
            vocab.load( file_path )
            self.encoder[ file_path.stem ] = vocab


    def encode_user_id(self, user_id: List[str] ) -> List[int]:
        """
            [Hash 1, Hash 2, Hash 3] -> [1, 2, 3]
        """
        return self.encoder['user'].encode(user_id)

    def encode_gender(self, gender: List[str]) -> List[int]:
        """
            ["female", "male", "other"] --> [2, 3, 4]
        """
        return self.encoder['gender'].encode(gender)

    def encode_titles(self, titles: List[str]) -> List[List[int]]:
        """
            titles: [ "藝文設計,服務業", "科技業" ]
                --> [ [id1, id2],       [id3] ]
        """
        return [ self.encoder['title'].encode(x.split(',')) for x in titles ]

    def encode_interests(self, interests: List[str]) -> Tuple[ List[List[int]], List[List[int]] ]:
        """
            interests: [ "藝術_角色設計,藝術_電腦繪圖", "藝術_繪畫與插畫" ]
                --> group:     [ ["藝術", "藝術"],        ["藝術"] ]
                    subgroups: [ ["角色設計", "電腦繪圖"], ["繪畫與插圖"] ]
                --> group:     [ [7, 7], [7] ]
                    subgroups: [ [3, 4], [6] ]

        """
        groups, subgroups = [], []
        for user_interests in interests:
            # user_interests = 'G1_SG1,G2_SG2'
            data = [ (g, sg) for (g, sg) in self.interest_generator(user_interests) ]
            groups.append( self.encoder['group'].encode([ g for g, _ in data ]) )
            subgroups.append( self.encoder['subgroup'].encode([ sg for _, sg in data ]) )
        return groups, subgroups
        # return [ 
        #     (self.encoder['group'].token_to_id(group), self.encoder['subgroup'].token_to_id(subgroup) )
        #     for x in interests
        #     for (group, subgroup) in self.interest_generator(x)
            
        # ]
    def encode_recreations(self, recreations: List[str]) -> List[List[int]]:
        """
            recreations: [ "睡覺,電玩", "吸貓" ] --> [ [id1, id2], [id3] ]
        """
        return [ self.encoder['recreation'].encode(x.split(',')) for x in recreations ]

class BertUserPreprocessor( BasicUserPreprocessor ):
    def __init__(self, vocab_dir: Union[str, Path], column_names, pretrained_name):
        super().__init__(vocab_dir, column_names)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self.encoder['gender'] = self.tokenizer
        self.encoder['title'] = self.tokenizer
        self.encoder['group'] = self.tokenizer
        self.encoder['subgroup'] = self.tokenizer
    
    def encode_sentences(self, sents: List[str]) -> List[ List[int] ]:
        res = self.tokenizer(
            text=sents,
            truncation=True,
            max_length=256,
        )
        return res['input_ids']

    def encode_gender(self, gender: List[str]) -> List[int]:
        return self.encode_sentences(gender)

    def encode_titles(self, titles: List[str]) -> List[List[int]]:
        return self.encode_sentences(titles)

    def encode_interests(self, interests: List[str]) -> Tuple[ List[List[int]], List[List[int]] ]:
        groups, subgroups = [], []
        for user_interests in interests:
            # user_interests = 'G1_SG1,G2_SG2'
            data = [ (g, sg) for (g, sg) in self.interest_generator(user_interests) ]
            group_str = ','.join([ g for g, _ in data ])
            subgroup_str = ','.join([ sg for _, sg in data ])
            groups.append( group_str )
            subgroups.append( subgroup_str )
        return self.encode_sentences(groups), self.encode_sentences(subgroups)

    def encode_recreations(self, recreations: List[str]) -> List[List[int]]:
        return self.encode_sentences(recreations)
    
def prepare_user_datasets( users_dataset: Dataset, user_p: UserPreprocessor, batch_size: int, with_bos_eos: bool ) -> Dataset:
    D = users_dataset.map(
        user_p.append_bos_eos if with_bos_eos else user_p.fill_none_as_unk,
        batch_size=batch_size,
        batched=True,
    )
    D = D.map(
        user_p.encode,
        batch_size=batch_size,
        batched=True,
    )
    D = D.remove_columns(['interests'])
    D = D.sort("user_id")
    return D

if __name__ == "__main__":
    
    user_data = datasets.Dataset.from_csv( "../../data/users.csv" )
    user_p = BertUserPreprocessor(
        "../../cache/vocab",
        column_names=user_data.column_names,
        pretrained_name="bert-base-multilingual-cased"
    )
    batch_size = 32

    for flag in [False]:
        X = user_data.select(range(5))
        Y = prepare_user_datasets( X, user_p, batch_size, flag )
        
        print()
        for column in X.column_names:
            if column == 'interests': continue
            print(f"[{column}]:")
            print(f"  {X[column]} ->  {Y[column]}")

        print(f"[interests]:")
        print(f"  {X['interests']}")
        print(f"  {Y['groups']}")
        print(f"  {Y['subgroups']}")
        print()

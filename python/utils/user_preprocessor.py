from interface import Preprocessor
from typing import *
from pathlib import Path
import json
from vocab import Vocab
import datasets
from datasets import Dataset

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
        self.__none_value__ = [Vocab.UNK, Vocab.UNK, Vocab.UNK, f"{Vocab.UNK}_{Vocab.UNK}", Vocab.UNK]

    @staticmethod
    def interest_generator(interests: str):
        for interest in interests.split(','):
            group, subgroup = interest.split('_')
            yield (group, subgroup)

    def fill_none(self, batch: dict):
        for column, value in zip(self.column_names, self.__none_value__):
            batch[column] = [
                data if data != None else value
                for data in batch[column]
            ]
        return batch

    def encode(self, user_profile):
        res = {
            key: func( user_profile[key] ) for key, func in zip(self.column_names, self.__encode_func__)
        }
        interest_col = self.column_names[-2]
        res['groups'] = res[ interest_col ][0]
        res['subgroups'] = res[ interest_col ][1]
        del res[ interest_col ]
        return res

    def encode_user_id(self, user_id: str):
        raise NotImplementedError
    def encode_gender(self, gender: str):
        raise NotImplementedError
    def encode_titles(self, titles: str):
        raise NotImplementedError
    def encode_interests(self, interests: str):
        raise NotImplementedError
    def encode_recreations(self, recreations: str):
        raise NotImplementedError



class BasicUserPreprocessor( UserPreprocessor ):
    """
        mapping all attribute into idx
        ex: 54f305134ec3c809002e4ab7,female,自由業,"生活品味_靈性發展,生活品味_寵物","狗派,貓派"]
        ->  [ 1, 1, [2], [(3,4),(3,5)], [1,2] ]
    """
    def __init__(self, vocab_dir: Union[str, Path], column_names):
        """
            config_dir: 5 json files for converting each attributes into idx.
                ex: user_idx.json ...
        """
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
        return self.encoder['user'].encode(user_id)

    def encode_gender(self, gender: List[str]) -> List[int]:
        return self.encoder['gender'].encode(gender)

    def encode_titles(self, titles: List[str]) -> List[List[int]]:
        # Titles = "title1,title2,title3"
        return [ self.encoder['title'].encode(x.split(',')) for x in titles ]

    def encode_interests(self, interests: List[str]) -> Tuple[ List[List[int]], List[List[int]] ]:
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
    def encode_recreations(self, recreations: str) -> List[int]:
        return [ self.encoder['recreation'].encode(x.split(',')) for x in recreations ]


def prepare_user_datasets( users_dataset: Dataset, user_p: UserPreprocessor, batch_size: int ) -> Dataset:
    D = users_dataset.map(
        user_p.fill_none,
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
    user_p = BasicUserPreprocessor("../../cache/vocab", column_names=user_data.column_names)
    batch_size = 32

    user_data = user_data.select(range(1000))
    user_data = prepare_user_datasets( user_data, user_p, batch_size )

    print(user_data[:10])

    idx = 13
    print(user_data[idx])

import json
from typing import *
from pathlib import Path
import argparse


class Vocab:
    PAD = "[PAD]"
    UNK = "[UNK]"

    def __init__(self, vocab: Iterable[str] = [], with_sepcial_token=True):
        if with_sepcial_token:
            self.token2idx = {
                Vocab.PAD: 0,
                Vocab.UNK: 1,
                **{token: i for i, token in enumerate(vocab, 2)},
            }
        else:
            self.token2idx = {token: i for i, token in enumerate(vocab)}

    @property
    def token_num(self):
        return len(self.token2idx)
    @property
    def pad_id(self) -> int:
        if Vocab.PAD in self.token2idx:
            return self.token2idx[Vocab.PAD]
        else:
            return -1
    @property
    def unk_id(self) -> int:
        if Vocab.UNK in self.token2idx:
            return self.token2idx[Vocab.UNK]
        else:
            return -1

    @property
    def tokens(self) -> List[str]:
        return list(self.token2idx.keys())
    
    def add_token(self, token):
        if token in self.token2idx:
            return

        self.token2idx[token] = self.token_num

    def token_to_id(self, token: str) -> int:
        return self.token2idx.get(token, self.unk_id)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id(token) for token in tokens]

    def encode_batch(
        self, batch_tokens: List[List[str]], to_len: int = None
    ) -> List[List[int]]:
        batch_ids = [self.encode(tokens) for tokens in batch_tokens]
        to_len = max(len(ids) for ids in batch_ids) if to_len is None else to_len
        padded_ids = self.pad_to_len(batch_ids, to_len, self.pad_id)
        return padded_ids

    @staticmethod
    def pad_to_len(seqs: List[List[int]], to_len: int, padding: int) -> List[List[int]]:
        paddeds = [seq[:to_len] + [padding] * max(0, to_len - len(seq)) for seq in seqs]
        return paddeds

    def save(self, json_file: Union[str, Path]):
        with open(json_file, 'w') as f:
            json.dump(self.token2idx, f, indent=2, ensure_ascii=False)

    def load(self, json_file: Union[str, Path]):
        with open(json_file, 'r') as f:
            self.token2idx = json.load(f)

def parse()  -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument( "--data_dir", type=str, default='../../data' )
    parser.add_argument( "--output_dir", type=str, default='../../cache/vocab')

    args = parser.parse_args()

    p = Path(args.data_dir)
    assert p.exists(), f"data_dir {p} doesn't exist"

    return args



def main():
    # User:
    #     - user_id
    #     - gender: ['male', 'female', 'others']
    #     - occupation_titles: "藝文設計,服務業"
    #     - interests: "藝術_角色設計,藝術_電腦繪圖,藝術_繪畫與插畫,設計_平面設計"
    #     - recreation_names: "手作,手寫字,插畫,水彩,貓派,速寫,閱讀,電影,電玩,電腦繪圖"
    # Course:
    #     * course_id: hash str
    #     * teacher_id: hash str
    #     * groups: 課程分類,逗號分隔多項. ex: '程式,行銷','程式,設計','職場技能'
    #     * sub_groups: 課程子分類，使用逗號分隔多項. ex: "更多語言,歐洲語言"
    #     * topics: 課程主題,逗號分隔多項. 'PowerPoint,簡報技巧'
    import pandas as pd
    import numpy as np

    args = parse()
    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    user_df = pd.read_csv( data_dir / "users.csv" )
    course_df = pd.read_csv( data_dir / "courses.csv" )
    subgroup_df = pd.read_csv( data_dir / "subgroups.csv" )

    def flatten( l : List[List]):
        return [item for sublist in l for item in sublist]

    def dump_mapping( df: pd.DataFrame, column_name, output_name, sep = None, with_special_tokens=True):
        data = np.unique(df[column_name].fillna(Vocab.UNK))

        if sep:
            data = [ item.split(sep) for item in data ]
            data = np.unique(flatten(data))
        
        data = data[ data != Vocab.UNK ]
        vocab = Vocab(data, with_special_tokens)

        print("Dumping", output_name)
        print("Vocab Size:", vocab.token_num)
        print()
        vocab.save( output_dir / output_name )

    dump_mapping(user_df, 'user_id', 'user.json', with_special_tokens=False)
    dump_mapping(user_df, 'gender', 'gender.json')
    dump_mapping(user_df, 'occupation_titles', 'title.json', ',' )
    dump_mapping(user_df, 'recreation_names', 'recreation.json', ',' )

    dump_mapping(course_df, 'course_id', 'course.json')
    dump_mapping(course_df, 'teacher_id', 'teacher.json')
    dump_mapping(course_df, 'topics', 'topic.json', ',')

    
    # Group & Subgroup from User Interests
    user_group, user_subgroup = set(), set()
    for interests in user_df['interests'].fillna( f"{Vocab.UNK}_{Vocab.UNK}"):
        # print(interests)
        for interest in interests.split(','):
            group, subgroup = interest.split('_')
            # print(group, subgroup)
            user_group.add(group)
            user_subgroup.add(subgroup)
        

    # Group & Subgroup from Course
    course_group = set()
    course_subgroup = set()
    for groups in course_df['groups'].fillna(Vocab.UNK):
        for group in groups.split(','):
            course_group.add(group)
    for subgroups in course_df['sub_groups'].fillna(Vocab.UNK):
        for subgroup in subgroups.split(','):
            course_subgroup.add(subgroup)
    
    
    # Merge Group & Subgroup
    merged_group = sorted( user_group | course_group )
    merged_subgroup = sorted( set(subgroup_df['subgroup_name']) | user_subgroup | course_subgroup )

    merged_group.remove(Vocab.UNK)
    merged_subgroup.remove(Vocab.UNK)
    # Dump Group & subgroups
    print("Dumping Group & Subgroups")
    Vocab(merged_group).save( output_dir / 'group.json' )
    Vocab(merged_subgroup).save( output_dir / 'subgroup.json' )

    print("Notice: the subgroup mapping is different from the original one!")

    print("Finish")


if __name__ == "__main__":
    main()

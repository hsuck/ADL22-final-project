import pandas as pd
import json
from typing import *
from pathlib import Path

special_tokens = ['[UKN]']

def key_to_id_map( keys: List[Hashable] ):
    key2id = {}
    id2key = []
    for token in special_tokens:
        id = len(id2key)
        if token not in key2id:
            key2id[token] = id
            id2key.append(token)
        
    for k in keys:
        id = len(id2key)
        if k not in key2id:
            key2id[k] = id
            id2key.append(k)
    return key2id, id2key

def save_map( filename: str, obj: Union[Dict, List], indent: int = 2 ):
    with open( filename, 'w' ) as f:
        json.dump( obj, f, indent=indent, ensure_ascii=False )


if __name__ == "__main__":
    data_dir = Path("../data")
    cache_dir = Path("../cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    files = [
        data_dir / 'courses.csv',
        data_dir / 'users.csv',
        data_dir / 'subgroups.csv',
    ]
    outputs = [
        cache_dir / 'course_id.json',
        cache_dir / 'user_id.json',
        cache_dir / 'subgroups_id.json',
    ]


    for filename, output in zip(files, outputs):
        df = pd.read_csv(filename)
        columns = list(df.keys())

        id_column = columns[0]
        if filename.name == 'subgroups.csv':
            id_column = columns[1]

        k2i, i2k = key_to_id_map( df[ id_column ] )

        save_map(output, k2i)

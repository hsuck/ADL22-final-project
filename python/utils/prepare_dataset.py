from typing import *
from datasets import Dataset, DatasetDict
from pathlib import Path
from torch.utils.data import DataLoader
from torch import LongTensor, Tensor

from .user_preprocessor import BasicUserPreprocessor, prepare_user_datasets
from .topic_reorder import TopicReorder

def pad_to_max_len(seqs: List[List[int]], padding: int) -> List[List[int]]:
    max_len = max( [ len(s) for s in seqs] )
    paddeds = [seq[:max_len] + [padding] * (max_len - len(seq)) for seq in seqs]

    return paddeds

def prepare_course_datadict(seen: bool, data_dir: Path, vocab_dir: Path) -> DatasetDict:
    data_files = {
        'train': 'train.csv',
        'eval': 'val_seen.csv' if seen else 'val_unseen.csv',
        'test': 'test_seen.csv' if seen else 'test_unseen.csv',
    }

    users_csv, courses_csv = Path(data_dir) / 'users.csv', Path(data_dir) / 'courses.csv'
    data_files = {
        key: Path(data_dir) / Path(name) for key, name in data_files.items()
    }
    pass

def prepare_topic_datadict(seen: bool, data_dir: Path, vocab_dir: Path) -> DatasetDict:
    data_files = {
        'train': 'train_group.csv',
        'eval': 'val_seen_group.csv' if seen else 'val_unseen_group.csv',
        'test': 'test_seen_group.csv' if seen else 'test_unseen_group.csv',
    }

    users_csv= Path(data_dir) / 'users.csv'
    data_files = {
        key: Path(data_dir) / Path(name) for key, name in data_files.items()
    }

    user_data = Dataset.from_csv( str(users_csv) )
    user_p = BasicUserPreprocessor(vocab_dir, column_names=user_data.column_names)
    user_data = prepare_user_datasets( user_data, user_p, batch_size=32 )

    def user2feats(inputs):
        inputs['user_id'] = user_p.encode_user_id(inputs['user_id'])
        feats = user_data.select( inputs['user_id'] )
        for key in feats.column_names:
            if key == 'user_id': continue
            inputs[key] = feats[key]
        return inputs

    data_dict = {}
    TR = TopicReorder(vocab_dir, data_dir)
    for key, file_path in data_files.items():
        print(f"[{key}]: Processing {file_path}")
        ds = Dataset.from_csv( str(file_path) )
        if key != 'test':
            ds = TR.process_dataset(ds)
            ds = ds.rename_column('subgroup', 'target')
        else:
            ds = ds.remove_columns(['subgroup'])
        
        data_dict[key] = ds.map(
            user2feats,
            batch_size=32,
            batched=True,
        )


    return DatasetDict(data_dict)


def prepare_topic_dataloader(seen: bool, data_dir: Path, vocab_dir: Path, batch_size: int):
    data_dict = prepare_topic_datadict(seen, data_dir, vocab_dir )

    def collate_fn(inputs: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        column_names = inputs[0].keys()
        res = {}
        for key in column_names:
            padding = 0 if key != 'target' else -100
            res[key] = [ data[key] for data in inputs ]
            if type(res[key][0]) is list:
                res[key] = pad_to_max_len(res[key], padding)
            res[key] = LongTensor(res[key])
        return res
    
    loader_dict = {}
    for key, ds in data_dict.items():
        loader_dict[key] = DataLoader(
            ds,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=4,
            shuffle = (key == 'train'),
        )
    return loader_dict

if __name__ == "__main__":
    loaders = prepare_topic_dataloader(False, "../../data", "../../cache/vocab", 4 )
    print(loaders)

    for i, data in enumerate(loaders['train']):
        if i == 4: break
        for key, val in data.items():
            print(f"{key}:\n{val}")

        break
    print()

    for i, data in enumerate(loaders['test']):
        if i == 4: break
        for key, val in data.items():
            print(f"{key}: {val.shape}\n{val}")

        break
    print()
        
    

    pass
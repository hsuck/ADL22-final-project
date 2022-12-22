from typing import *
from datasets import Dataset, DatasetDict
from pathlib import Path
from torch.utils.data import DataLoader
from torch import LongTensor, Tensor

try:
    from .user_preprocessor import BasicUserPreprocessor, prepare_user_datasets
    from .course_preprocessor import CoursePreprocessor
    from .topic_reorder import TopicReorder
except:
    from user_preprocessor import BasicUserPreprocessor, prepare_user_datasets
    from course_preprocessor import CoursePreprocessor
    from topic_reorder import TopicReorder

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

    ## User Data
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

    ## [TODO]: Course Data
    course_data = Dataset.from_csv( str(users_csv) )
    course_p = CoursePreprocessor(vocab_dir, column_names=user_data.column_names)
    # course_data = prepare_course_dataset( course_data, course_p, batch_size=32 )

    def course2feats(inputs):
        raise NotImplementedError
        # inputs['course_id'] = course_p.encode_course_id(inputs['course_id'])
        # feats = user_data.select( inputs['course_id'] )
        # for key in feats.column_names:
        #     if key == 'user_id': continue
        #     inputs[key] = feats[key]
        # return inputs
    data_dict = {}
    for key, file_path in data_files.items():
        print(f"[{key}]: Processing {file_path}")
        ds = Dataset.from_csv( str(file_path) )
        if key != 'test':
            ds = ds.rename_column('course_id', 'target')
        else:
            ds = ds.remove_columns(['course_id'])
        
        data_dict[key] = ds.map(
            user2feats,
            batch_size=32,
            batched=True,
        )
        data_dict[key] = ds.map(
            course2feats,
            batch_size=32,
            batched=True,
        )


    return DatasetDict(data_dict)
    pass

def prepare_course_dataloader(seen: bool, data_dir: Path, vocab_dir: Path, batch_size: int) -> Dict[str, DataLoader]:
    data_dict = prepare_course_datadict(seen, data_dir, vocab_dir )

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

def prepare_topic_datadict(
        seen: bool,
        data_dir: Path,
        vocab_dir: Path,
        with_bos_eos: bool
    ) -> DatasetDict:

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
    
    user_data = prepare_user_datasets( user_data, user_p, batch_size=32, with_bos_eos=False )

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
            ds = TR.process_dataset(ds, with_bos_eos)
            ds = ds.rename_column('subgroup', 'target')
        else:
            ds = ds.remove_columns(['subgroup'])
        
        data_dict[key] = ds.map(
            user2feats,
            batch_size=32,
            batched=True,
        )


    return DatasetDict(data_dict)


def prepare_topic_dataloader(seen: bool, data_dir: Path, vocab_dir: Path, batch_size: int, with_bos_eos: bool = False) -> Dict[str, DataLoader]:
    """
        vocab_dir: refer to README
        with_bos_eos: the format of output, default is False
            False for classification model
            True for sequence output model ( input_seq = [bos, labels], target_seq = [labels, eos] )
    """
    data_dict = prepare_topic_datadict(seen, data_dir, vocab_dir, with_bos_eos )

    def collate_fn(inputs: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        column_names = inputs[0].keys()
        model_input = {}
        for key in column_names:
            if key == 'target': continue
            padding = 0 if key != 'target' else -100
            model_input[key] = [ data[key] for data in inputs ]
            if type(model_input[key][0]) is list:
                model_input[key] = pad_to_max_len(model_input[key], padding)
            model_input[key] = LongTensor(model_input[key])
        
        target = None
        if 'target' in column_names:
            ignore_idx = -100
            target = [ data['target'] for data in inputs ]
            if type(target) is list:
                target = pad_to_max_len(target, ignore_idx)
            target = LongTensor(target)

            if with_bos_eos:
                input_seq = LongTensor(target[:, :-1])
                target_seq = LongTensor(target[:, 1:])

                model_input['input_seq'] = input_seq
                target = target_seq
        
        return model_input, target
    
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
    loaders = prepare_topic_dataloader(False, "../../data", "../../cache/vocab", 4, False )
    print(loaders)

    for i, (model_input, target_seq) in enumerate(loaders['train']):
        if i == 4: break
        for key, val in model_input.items():
            print(f"{key}:\n{val}")
        print(f"target:\n{target_seq}")

        break
    print()

    for i, (model_input, target_seq) in enumerate(loaders['test']):
        if i == 4: break
        for key, val in model_input.items():
            print(f"{key}:\n{val}")
        print(f"target:\n{target_seq}")

        break
    print()

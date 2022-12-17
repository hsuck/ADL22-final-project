import torch
from functools import partial
from torch.utils.data import DataLoader
from prepare_dataset import prepare_topic_pred
from typing import List

def pad_to_max_len(seqs: List[List[int]], padding: int) -> List[List[int]]:
    max_len = max( [ len(s) for s in seqs] )
    paddeds = [seq[:max_len] + [padding] * (max_len - len(seq)) for seq in seqs]

    return paddeds

def topic_collate_fn(inputs, user_data):
    # Inputs
    padding_id = 0
    user_ids = [ data['user_id'] for data in inputs]
    gender = [ user_data['gender'][id] for id in user_ids ]
    titles = pad_to_max_len( [ user_data['occupation_titles'][id] for id in user_ids ], padding_id )
    recreations = pad_to_max_len( [ user_data['recreation_names'][id] for id in user_ids ], padding_id )
    groups = pad_to_max_len( [ user_data['groups'][id] for id in user_ids ], padding_id )
    subgroups = pad_to_max_len( [ user_data['subgroups'][id] for id in user_ids ], padding_id )


    # Target outputs
    target = [ data['subgroup'] for data in inputs]
    ignore_idx = -100 # for loss computing
    target = pad_to_max_len(target, ignore_idx)
    return {
        'user_id': torch.LongTensor(user_ids),
        'gender': torch.LongTensor(gender),
        'titles': torch.LongTensor(titles),
        'recreations': torch.LongTensor(recreations),
        'groups': torch.LongTensor(groups),
        'subgroups': torch.LongTensor(subgroups),
        'target': torch.LongTensor(target),
    }

if __name__ == "__main__":
    data_dict = prepare_topic_pred(False, "../../data", "../../cache/vocab" )
    print(data_dict)
    for key, ds in data_dict.items():
        print(f"[{key}]: ")
        print(f"    {ds[:5]}" )

    # print()
    # print(subgroup_data)
    # print(subgroup_data[:5])
    # eval_dataloader = DataLoader(
    #     data_dict['eval'],
    #     collate_fn=partial(topic_collate_fn, user_data=user_data),
    #     batch_size=2
    # )
    
    # print()
    # cnt = 4
    # for data in eval_dataloader:
    #     print(data)
    #     if cnt == 0: break
    #     cnt -= 1
    # pass
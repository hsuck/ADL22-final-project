from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import BertForSequenceClassification, get_scheduler, AutoTokenizer
from typing import Dict, List, Any, Tuple
import numpy as np
from tqdm import tqdm
import math

from accelerate import Accelerator

import torch
from torch.utils.data import DataLoader
from torch import LongTensor

from functools import partial
import json
from utils.user_preprocessor import User2SeqPreprocessor, prepare_user_datasets
import ml_metrics

#####################
### Global Variables
#####################

gradient_accumulation_steps = 2
num_warmup_steps = 0
learning_rate = 5e-4
NUM_EPOCH = 10
pretrained_model_name="bert-base-multilingual-uncased"
batch_size = 4

data_dir = Path("../data")
vocab_dir = Path("../cache/vocab")
output_dir = Path(f"../output/{pretrained_model_name}")

SEEN_COURSE = False
DEVICE = 'cpu'  # depends on accelerate.device
DEBUG = False   # only use 100 train/eval/test data
PAD_TOKEN = '[PAD]'
PAD_TOKEN_ID = 0
NUM_OF_CLASS = -1

METRIC_LOG_CSV = output_dir / 'metric_log.csv'


#####################
### utils
#####################
def get_course2id():
    return json.load( (vocab_dir/'course.json').open() )

def get_id2course():
    c2id = json.load( (vocab_dir/'course.json').open() )
    return {
        id: cid for cid, id in c2id.items()
    }

def get_csv_filename( seen: bool ):
    data_files = {
        'train': 'train.csv',
        'eval': 'val_seen.csv' if seen else 'val_unseen.csv',
        'test': 'test_seen.csv' if seen else 'test_unseen.csv',
    }
    return data_files

#####################
### Dataset
#####################

def prepare_datasets() -> Tuple[Dataset, DatasetDict]:
    global PAD_TOKEN, PAD_TOKEN_ID, NUM_OF_CLASS, data_dir, vocab_dir

    user_data = Dataset.from_csv( str( data_dir / 'users.csv') )
    user_p = User2SeqPreprocessor(
        vocab_dir=vocab_dir,
        column_names=user_data.column_names,
        pretrained_name=pretrained_model_name
    )
    user_ds: Dataset = prepare_user_datasets(user_data, user_p, batch_size=32, with_bos_eos=False)
    user_ds = user_ds.remove_columns( [col for col in user_ds.column_names if col not in ['user_id', 'model_inputs', 'input_str']])

    user_ds.sort(column='user_id')

    train_data = Dataset.from_csv( str( data_dir / 'train.csv') )
    eval_data = Dataset.from_csv( str( data_dir / 'val_unseen.csv') )
    test_data = Dataset.from_csv( str( data_dir / 'test_unseen.csv') )

    PAD_TOKEN = user_p.tokenizer.pad_token
    PAD_TOKEN_ID = user_p.tokenizer.pad_token_id

    if DEBUG:
        train_data = train_data.select( range(100) )
        eval_data = eval_data.select( range(100) )
        test_data = test_data.select( range(100) )
    
    ds_dict: Dict[str, Dataset] = {
        'train': train_data,
        'eval': eval_data,
        'test': test_data,
    }

    course2id = get_course2id()
    NUM_OF_CLASS = len(course2id)

    for key, ds in ds_dict.items():
        idxs = user_p.encode_user_id( ds['user_id'] )
        input_str = user_ds.select(idxs)['input_str']
        ds_dict[key] = ds_dict[key].add_column('input_str', input_str)

        if key != 'test':
            label = [
                [course2id[x] for x in courses.split(' ') ] if courses != None
                else [0]
                for courses in ds['course_id']
            ]
            ds_dict[key] = ds_dict[key].add_column('label', label)
        
        ds_dict[key] = ds_dict[key].remove_columns('course_id')

    
    # print(user_p.tokenizer.pad_token)
    # print(user_p.tokenizer.pad_token_id)
    return DatasetDict(ds_dict)


def collate_fn(inputs, mode):
    input_str = [ x['input_str'] for x in inputs ]
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    model_inputs = tokenizer(
        input_str,
        padding=True,
        truncation=True,
        max_length=512,
    )
    for key, val in model_inputs.items():
        model_inputs[key] = LongTensor(val)
    
    sparse_labels = None
    labels = None
    if mode != 'test':
        sparse_labels = [ x['label'] for x in inputs ]
        labels = np.zeros( (len(inputs), NUM_OF_CLASS), dtype=np.int32 )
        for i, x in enumerate(inputs):
            labels[i, x['label']] = 1
            
        labels = torch.FloatTensor(labels)

    return model_inputs, labels, sparse_labels


#####################
### Training Process
#####################
def main():

    ds_dict = prepare_datasets()

    print(ds_dict)
    print(ds_dict['train'][:1])

    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    DEVICE = accelerator.device
    print(accelerator.state)

    train_dl = DataLoader(
        ds_dict['train'],
        batch_size=batch_size,
        collate_fn=partial( collate_fn, mode='train'),
        shuffle=True,
    )
    eval_dl = DataLoader(
        ds_dict['eval'],
        batch_size=batch_size,
        collate_fn=partial( collate_fn, mode='eval'),
        shuffle=False,
    )
    test_dl = DataLoader(
        ds_dict['test'],
        batch_size=batch_size,
        collate_fn=partial( collate_fn, mode='test'),
        shuffle=False,
    )

    # for x, y, sparse_y in train_dl:
    #     for key, val in x.items():
    #         print(f"x[{key}]:")
    #         print(val)
    #     print( np.where(y.cpu().numpy()==1))
    #     print( sparse_y)
    #     break
    # input()


    model = BertForSequenceClassification.from_pretrained(
            pretrained_model_name,
            num_labels=NUM_OF_CLASS,
            problem_type="multi_label_classification"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    num_update_steps_per_epoch = math.ceil(len(train_dl) / gradient_accumulation_steps)
    max_train_steps = NUM_EPOCH * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name= 'linear',
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps  * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    model.to(DEVICE)

    model, optimizer, train_dl, eval_dl, test_dl, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dl, eval_dl, test_dl, lr_scheduler
    )

    def make_prediction( logits: torch.Tensor, k=50 ):
        topk = torch.topk(logits, k=k, dim=-1)
        return topk.indices

    def train_step( dataloader ):
        model.train()
        total_loss = 0
        progress = tqdm( total = len(dataloader))
        for x, y, _ in dataloader:
            o = model( **x, labels=y )
            loss = o.loss
            total_loss += loss.detach().float()

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress.update()

        return {
            'loss': total_loss / len(dataloader)
        }

    def eval_step( dataloader: DataLoader ):
        model.eval()

        total_loss = 0
        mAP = 0
        progress = tqdm( total = len(dataloader) )
        for x, y, sparse_y in dataloader:
            with torch.no_grad():
                o = model( **x, labels=y )
            loss = o.loss

            total_loss += loss.detach().float()
            pred = make_prediction(o.logits).detach().cpu()
            mAP += ml_metrics.mapk(sparse_y, pred, k=50) * len(pred)
            progress.update()
        

        return {
            'loss': total_loss / len(dataloader),
            'mAP': mAP / len(dataloader.dataset)
        }


    METRIC_LOG_FP = METRIC_LOG_CSV.open('w')
    METRIC_LOG_FP.write("epoch,train_loss,eval_loss,eval_mAP@50\n")
    for i in range(NUM_EPOCH):
        train_metric = train_step( train_dl )
        eval_metric = eval_step( eval_dl )

        print(f"Epoch [{i:2d}]:")
        for key, val in train_metric.items():
            print(f"  train {key:5s}: {val:.6f}")
        for key, val in eval_metric.items():
            print(f"  eval  {key:5s}: {val:.6f}")
        
        METRIC_LOG_FP.write(f"{i},{train_metric['loss']:.6f},{eval_metric['loss']:.6f},{eval_metric['mAP']:.6f}\n")
        

    METRIC_LOG_FP.close()

    print(f"Saving model to {output_dir}")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
    )

    ### Inference
    def inference( output_file: Path, dataloader: DataLoader, uid_list: List[str] ):
        print(f"Inference {output_file}")

        assert len(dataloader.dataset) == len(uid_list)
        progress = tqdm(total=len(dataloader))
        id2course = get_id2course()
        fp = output_file.open('w')
        fp.write('user_id,course_id\n')
        idx = 0
        for x, _, _ in dataloader:
            with torch.no_grad():
                o = model( **x )
            pred = make_prediction(o.logits).detach().cpu().numpy()
            
            for p in pred:
                uid = uid_list[idx]
                s = ' '.join([ id2course[x] for x in p ])
                fp.write(f'{uid},{s}\n')
                idx += 1
            
            progress.update()
    
    inference(output_dir / 'eval_pred.csv', eval_dl, ds_dict['eval']['user_id'] )
    inference(output_dir / 'test_pred.csv', test_dl, ds_dict['test']['user_id'] )
            

        
        

if __name__ == "__main__":
    main()
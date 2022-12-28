import torch
import pandas as pd
import numpy as np
import os
import json
import csv
import datasets
import random
import collections
from collections import OrderedDict, Counter
from typing import *
from pathlib import Path
from datasets import Dataset

from utils.user_preprocessor import BasicUserPreprocessor, prepare_user_datasets
from utils.course_preprocessor import BasicCoursePreprocessor, prepare_course_datasets, course_item_features
from utils.average_precision import *
from torch_rechub.utils.data import pad_sequences, MatchDataGenerator
from torch_rechub.utils.match import generate_seq_feature_match, gen_model_input, negative_sample, Annoy
from torch_rechub.basic.features import SparseFeature, SequenceFeature, DenseFeature
from torch_rechub.utils.data import df_to_dict
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer
from torch_rechub.basic.metric import topk_metrics

os.environ['TOKENIZERS_PARALLELISM'] = "false"
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)
seed = 9527
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed( seed )
np.random.seed( seed )
random.seed( seed )

user_data = datasets.Dataset.from_csv("../data/users.csv")
user_p = BasicUserPreprocessor("../cache/vocab", column_names=user_data.column_names)
batch_size = 32

user_profile = prepare_user_datasets( user_data, user_p, batch_size, False )
useless = [
    'interests',
    # 'groups',
]
user_profile = user_profile.remove_columns(useless)

user_df = pd.DataFrame( user_profile, columns = user_profile.column_names )
user_df.rename( columns = { 'groups': 'groups_user' }, inplace = True )
user_df.rename( columns = { 'subgroups': 'sub_groups_user' }, inplace = True )

need_padding = [ 'occupation_titles', 'recreation_names', 'groups_user', 'sub_groups_user' ]
# need_padding = [ 'occupation_titles', 'recreation_names', 'interests' ]
for col in need_padding:
    user_df[col] = pad_sequences( user_df[col], maxlen = 50, value = 0, padding = 'pre', truncating = 'pre' ).tolist()
# print( user_df[:3] )
# input('>')

course_data = Dataset.from_csv( "../data/courses.csv" )
item_feat = course_item_features("../data/course_chapter_items.csv" )
course_p = BasicCoursePreprocessor("../cache/vocab",pretrained_name="bert-base-multilingual-cased", column_names=course_data.column_names)
batch_size = 32

course_profile = prepare_course_datasets( course_data, item_feat, course_p, batch_size, False )
useless =  [
    'course_name',
    'teacher_intro',
    'description',
    'will_learn',
    'required_tools',
    'recommended_background',
    # 'course_published_at_local',
    'unit_cnt',
    'assignment_cnt',
    # 'total_sec',
    'chapter_cnt',
    'target_group'
]
course_profile = course_profile.remove_columns( useless )

course_df = pd.DataFrame( course_profile, columns = course_profile.column_names )
# print( course_df )
course_df = course_df.sort_values('course_id')
course_df['course_id'] = ( np.array( course_df['course_id'] ) - 4 ).tolist()
# print( course_df )
# input('>')

# course_df['course_price'] = [
#     price // 100 for price in course_df['course_price']
# ]
# course_df.rename( columns = { 'groups': 'groups_course' }, inplace = True )

course_published_at_local = np.array( course_df['course_published_at_local'], dtype = float )
course_published_at_local = ( course_published_at_local - min( course_published_at_local ) ) / ( max( course_published_at_local ) - min( course_published_at_local ) )
# course_published_at_local = 2. * course_published_at_local - 1
course_df['course_published_at_local'] = course_published_at_local.tolist()

total_sec = np.array( course_df['total_sec'], dtype = float )
total_sec = ( total_sec - min( total_sec ) ) / ( max( total_sec ) - min( total_sec ) )
# total_sec = 2. * total_sec - 1
course_df['total_sec'] = total_sec.tolist()

course_price = np.array( course_df['course_price'], dtype = float )
course_price = ( course_price - min( course_price ) ) / ( max( course_price ) - min( course_price ) )
# course_price = 2. * course_price - 1
course_df['course_price'] = course_price.tolist()

# need_padding = [ 'groups_course', 'sub_groups', 'topics' ]
need_padding = [ 'groups', 'sub_groups', 'topics' ]
for col in need_padding:
    course_df[col] = pad_sequences( course_df[col], maxlen = 50, value = 0, padding = 'pre', truncating = 'pre' ).tolist()
# print( course_df[:3] )
# input('>')

train_data = pd.read_csv("../data/train.csv")
train_data['user_id'] = user_p.encode_user_id( train_data['user_id'] )

encoded_course_id = []
for hist in train_data['course_id']:
    encoded_course_id.append( ( np.array( course_p.encode_course_id( hist.split(' ') ) ) - 4 ).tolist() )

train_data['course_id'] = encoded_course_id
# print( train_data[:3] )
# input('>')

train_set = []
neg_idx = 0
items_cnt = Counter( [ course for hist in train_data['course_id'] for course in hist ] )
items_cnt_order = OrderedDict(sorted((items_cnt.items()), key=lambda x: x[1], reverse=True))
neg_list = negative_sample(items_cnt_order, ratio=139608 * 10, method_id=1)
for i in range( len( train_data ) ):
    uid = train_data['user_id'][i]
    pos_list = train_data['course_id'][i]
    for i in range( 1, len( pos_list ) ):
        hist_item = pos_list[:i]
        sample = [ uid, pos_list[i], hist_item, len( hist_item ) ]
        if i != len( pos_list ) - 1:
            last_col = 'label'
            train_set.append( sample + [1] )
            for _ in range( 3 ):
                sample[1] = neg_list[neg_idx]
                neg_idx += 1
                train_set.append( sample + [0] )

random.shuffle(train_set)
df_train = pd.DataFrame( train_set, columns = [ 'user_id', 'course_id', "hist_course_id", 'histlen_course_id' ] + [ last_col ] )

x_train = gen_model_input( df_train, user_df, 'user_id', course_df, 'course_id', seq_max_len = 50 )
y_train = x_train["label"]

# print({k: v[:3] for k, v in x_train.items()})
# input('>')

val_data = pd.read_csv("../data/val_seen.csv")
val_data['user_id'] = user_p.encode_user_id( val_data['user_id'] )

encoded_course_id = []
for hist in val_data['course_id']:
    encoded_course_id.append( ( np.array( course_p.encode_course_id( hist.split(' ') ) ) - 4 ).tolist() )

val_data['course_id'] = encoded_course_id

val_set = []
for i in range( len( val_data ) ):
    uid = val_data['user_id'][i]
    hist = val_data['course_id'][i]
    sample = [ uid, hist[len(hist) - 1], hist, len( hist ) ]
    last_col = 'label'
    val_set.append( sample + [1] )

# random.shuffle(val_set)
df_val = pd.DataFrame( val_set, columns = [ 'user_id', 'course_id', "hist_course_id", 'histlen_course_id' ] + [ last_col ] )

x_val = gen_model_input( df_val, user_df, 'user_id', course_df, 'course_id', seq_max_len = 50 )
y_val = x_val["label"]

# print({k: v[:3] for k, v in x_val.items()})
# input('>')
val_user = x_val

test_data = pd.read_csv("../data/test_seen.csv")
test_data['user_id'] = user_p.encode_user_id( test_data['user_id'] )

encoded_course_id = []
for hist in test_data['course_id']:
    encoded_course_id.append( ( np.array( course_p.encode_course_id( hist.split(' ') ) ) - 4 ).tolist() )

test_data['course_id'] = encoded_course_id

test_set = []
for i in range( len( test_data ) ):
    uid = test_data['user_id'][i]
    hist = test_data['course_id'][i]
    sample = [ uid, hist[len(hist) - 1], hist, len( hist ) ]
    last_col = 'label'
    test_set.append( sample + [1] )

df_test = pd.DataFrame( test_set, columns = [ 'user_id', 'course_id', "hist_course_id", 'histlen_course_id' ] + [ last_col ] )

x_test = gen_model_input( df_test, user_df, 'user_id', course_df, 'course_id', seq_max_len = 50 )
y_test = x_test["label"]

# print({k: v[:3] for k, v in x_test.items()})
test_user = x_test

embed_size = 64
user_features = [
    SparseFeature( 'user_id', vocab_size = user_df['user_id'].max() + 1, embed_dim = embed_size ),
    SparseFeature( 'gender', vocab_size = user_df['gender'].max() + 1, embed_dim = embed_size ),
]
user_features += [
    SequenceFeature("hist_course_id",
                    vocab_size = course_df["course_id"].max() + 1,
                    embed_dim = embed_size,
                    pooling = "mean",
                    shared_with = "course_id"),
    # SequenceFeature("interests",
    #                 vocab_size = 99,
    #                 embed_dim = embed_size,
    #                 pooling = "mean"),
    #                 # shared_with = "course_id"),
    SequenceFeature("groups_user",
                    vocab_size = 16,
                    embed_dim = embed_size,
                    pooling = "mean"),
                    # shared_with = "groups_course"),
    SequenceFeature("sub_groups_user",
                    vocab_size = 99,
                    embed_dim = embed_size,
                    pooling = "mean"),
                    # shared_with = "sub_groups"),
    SequenceFeature("occupation_titles",
                    vocab_size = 24,
                    embed_dim = embed_size,
                    pooling = "mean"),
                    # shared_with = "course_id"),
    SequenceFeature("recreation_names",
                    vocab_size = 35,
                    embed_dim = embed_size,
                    pooling = "mean"),
                    # shared_with = "course_id"),
]

course_features = [
    SparseFeature( 'course_id', vocab_size = course_df['course_id'].max() + 1, embed_dim = embed_size ),
    # SparseFeature( 'course_price', vocab_size = course_df['course_price'].max() + 1, embed_dim = embed_size ),
    SparseFeature( 'teacher_id', vocab_size = course_df['teacher_id'].max() + 1, embed_dim = embed_size ),
    # SparseFeature( 'course_published_at_local', vocab_size = course_df['course_published_at_local'].max() + 1, embed_dim = embed_size ),
    # SparseFeature( 'target_group', vocab_size = course_df['target_group'].max() + 1, embed_dim = embed_size ),
    # SparseFeature( 'chapter_cnt', vocab_size = course_df['chapter_cnt'].max() + 1, embed_dim = embed_size ),
]
course_features += [
    SequenceFeature("groups",
                    vocab_size = 16,
                    embed_dim = embed_size,
                    pooling = "mean",
                    shared_with = "groups_user"),
    SequenceFeature("sub_groups",
                    vocab_size = 99,
                    embed_dim = embed_size,
                    pooling = "mean",
                    shared_with = "sub_groups_user"),
    SequenceFeature("topics",
                    vocab_size = 205,
                    embed_dim = embed_size,
                    pooling = "mean"),
]
course_features += [
    DenseFeature('course_price'),
    DenseFeature('course_published_at_local'),
    DenseFeature('total_sec'),
]
print( user_features )
print( course_features )
# input('>')

all_item = df_to_dict( course_df )

dg = MatchDataGenerator( x = x_train, y = y_train )
train_dl, val_dl,  item_dl = dg.generate_dataloader( x_val, all_item, batch_size = 4096 )
train_dl, test_dl,  item_dl = dg.generate_dataloader( x_test, all_item, batch_size = 4096 )

model = DSSM( user_features,
              course_features,
              temperature = 1,
              user_params = {
                                "dims": [256, 128, 64],
                                "activation": 'prelu',  # important!!
                                # "dropout": 0.2,
              },
              item_params = {
                                "dims": [256, 128, 64],
                                "activation": 'prelu',  # important!!
                                # "dropout": 0.2,
              } )
print( model )
trainer = MatchTrainer( model,
                        mode=0,  # 同上面的mode，需保持一致
                        optimizer_params = { "lr": 1e-4,
                                             "weight_decay": 1e-6 },
                        # scheduler_fn = torch.optim.lr_scheduler.StepLR,
                        # scheduler_params = { "step_size": 15,
                        #                      "gamma": 0.5 },
                        n_epoch = 10,
                        device = 'cuda',
                        model_path = './' )
trainer.fit( train_dl )

def match_evaluation(user_embedding, item_embedding, test_user, all_item, user_col='user_id', item_col='course_id', topk=10):
    print("evaluate embedding matching on test data")
    annoy = Annoy(n_trees=50)
    annoy.fit(item_embedding)

    #for each user of test dataset, get ann search topk result
    print("matching for topk")

    preds = []
    for user_id, user_emb in zip( test_user[user_col], user_embedding ):
        items_idx, items_scores = annoy.query(v=user_emb, n=topk)  #the index of topk match items
        preds.append( items_idx )
    # print( preds )

    print("compute MAP@K metrics")
    score = mapk( val_data['course_id'], preds, k = 50 )
    print( 'MAP@50:', score )

user_embedding = trainer.inference_embedding(model=model, mode="user", data_loader=val_dl, model_path='./')
item_embedding = trainer.inference_embedding(model=model, mode="item", data_loader=item_dl, model_path='./')
match_evaluation(user_embedding, item_embedding, val_user, all_item, topk=50)


def Test(user_embedding, item_embedding, test_user, all_item, user_col='user_id', item_col='course_id', topk=10):
    print("evaluate embedding matching on test data")
    annoy = Annoy(n_trees=50)
    annoy.fit(item_embedding)
    #for each user of test dataset, get ann search topk result
    print("matching for topk")

    with open('../cache/vocab/user.json') as user_file:
        user_map= json.load(user_file)
    with open('../cache/vocab/course.json') as item_file:
        item_map= json.load(item_file)

    f = open('answer.csv' , 'w')
    writer = csv.writer(f)
    writer.writerow(['user_id', 'course_id'])

    user_key_list = list(user_map.keys())
    user_val_list = list(user_map.values())
    item_key_list = list(item_map.keys())
    item_val_list = list(item_map.values())
    for user_id, user_emb in zip(test_user[user_col], user_embedding):
        items_idx, items_scores = annoy.query(v=user_emb, n=topk)  #the index of topk match items
        # print(items_idx)
        position = user_val_list.index(user_id)
        ori_userid = user_key_list[position]
        ori_item = []
        for i in items_idx:
            position = item_val_list.index(i+4)
            ori_item.append(item_key_list[position])

        x = ' '.join(ori_item)
        ret = []
        ret.append(ori_userid)
        ret.append(x)
        writer.writerow(ret)

user_embedding = trainer.inference_embedding(model=model, mode="user", data_loader=test_dl, model_path='./')
item_embedding = trainer.inference_embedding(model=model, mode="item", data_loader=item_dl, model_path='./')
metrics = Test(user_embedding, item_embedding, test_user, all_item, topk=50)

import torch
import pandas as pd
import numpy as np
import os
import json
from utils.user_preprocessor import BasicUserPreprocessor, prepare_user_datasets
from utils.course_preprocessor import BasicCoursePreprocessor, prepare_course_datasets, course_item_features
from typing import *
from pathlib import Path
import json
import datasets
from datasets import Dataset

pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)
torch.manual_seed(2022)

user_data = datasets.Dataset.from_csv("../data/users.csv")
user_p = BasicUserPreprocessor("../cache/vocab", column_names=user_data.column_names)
batch_size = 32

user_profile = prepare_user_datasets( user_data, user_p, batch_size, False )
# print( user_profile[:3] )
user_df = pd.DataFrame( user_profile, columns = user_profile.column_names )
user_df["groups"] = user_df["groups"].apply(lambda x: x[0])
user_df["subgroups"] = user_df["subgroups"].apply(lambda x: x[0])
user_df["occupation_titles"] = user_df["occupation_titles"].apply(lambda x: x[0])
user_df["recreation_names"] = user_df["recreation_names"].apply(lambda x: x[0])
user_df.rename( columns = { 'groups': 'groups_x' }, inplace = True )
print( user_df[:3] )



course_data = Dataset.from_csv( "../data/courses.csv" )
item_feat = course_item_features("../data/course_chapter_items.csv" )
course_p = BasicCoursePreprocessor("../cache/vocab",pretrained_name="bert-base-multilingual-cased", column_names=course_data.column_names)
batch_size = 32

course_profile = prepare_course_datasets( course_data, item_feat, course_p, batch_size, False )
useless =  [ 'course_name', 'teacher_intro', 'description', 'will_learn', 'required_tools', 'recommended_background', 'target_group' ]
course_profile = course_profile.remove_columns( useless )
# print( course_profile[:3] )
course_df = pd.DataFrame( course_profile, columns = course_profile.column_names )
course_df["groups"] = course_df["groups"].apply(lambda x: x[0])
course_df["sub_groups"] = course_df["sub_groups"].apply(lambda x: x[0])
course_df["topics"] = course_df["topics"].apply(lambda x: x[0])
course_df.rename( columns = { 'groups': 'groups_y' }, inplace = True )
print( course_df[:3] )
#input('>')

train_data = pd.read_csv( "../data/train.csv" )
train_data['user_id'] = user_p.encode_user_id( train_data['user_id'] )

encoded_course_id = []
# histlen = []
for hist in train_data['course_id']:
    encoded_course_id.append( course_p.encode_course_id( hist.split(' ')  ) )
    # histlen.append( len( encoded_course_id[-1] ) )
train_data['course_id'] = encoded_course_id
# x_train = x_train.remove_columns( 'user_id' )
# x_train = x_train.add_column( 'user_id', encoded_user_id )
# x_train = x_train.remove_columns( 'course_id' )
# x_train = x_train.add_column( 'hist_course_id', encoded_course_id )
# x_train = x_train.add_column( 'histlen_course_id', histlen )

print( train_data[:3] )
# print( max( train_data['histlen_course_id'] ) )


train_set = []
neg_idx = 0
neg_list = np.random.choice( course_profile['course_id'], size = len( train_data['user_id'] ) * 3 * 155 , replace = True )
print( neg_list[:3] )
for i in range( len( train_data ) ):
    uid = train_data['user_id'][i]
    hist = train_data['course_id'][i]
    for i in range( 1, len( hist ) ):
        hist_item = hist[:i]
        sample = [ uid, hist[i], hist_item, len( hist_item ) ]
        last_col = 'label'
        train_set.append( sample + [1] )
        for _ in range( 3 ):
            sample[1] = neg_list[ neg_idx ]
            neg_idx += 1
            train_set.append( sample + [0] )
        # print( neg_idx )
        # input('>')
print( train_set[:5] )
df_train = pd.DataFrame( train_set, columns = [ 'user_id', 'course_id', "hist_course_id", 'histlen_course_id', 'label' ] )
print( df_train[:5] )
print( max( df_train['histlen_course_id'] ) )



from torch_rechub.utils.match import generate_seq_feature_match, gen_model_input
x_train = gen_model_input( df_train, user_df, 'user_id', course_df, 'course_id', seq_max_len = 50 )
y_train = x_train["label"]

print({k: v[:3] for k, v in x_train.items()})
print( y_train[:3] )
#input('>')

val_data = pd.read_csv( "../data/test_seen.csv")
val_data['user_id'] = user_p.encode_user_id( val_data['user_id'] )
encoded_course_id = []
for hist in val_data['course_id']:
    encoded_course_id.append( course_p.encode_course_id( hist.split(' ')  ) )
val_data['course_id'] = encoded_course_id
print( val_data[:3] )

val_set = []
neg_idx = 0
neg_list = np.random.choice( course_profile['course_id'], size = len( val_data['user_id'] ) * 3 * 155 , replace = True )
print( neg_list[:3] )
for i in range( len( val_data ) ):
    uid = val_data['user_id'][i]
    hist = val_data['course_id'][i]
    for i in range( 1, len( hist ) ):
        hist_item = hist[:i]
        sample = [ uid, hist[i], hist_item, len( hist_item ) ]
        last_col = 'label'
        val_set.append( sample + [1] )
        for _ in range( 3 ):
            sample[1] = neg_list[ neg_idx ]
            neg_idx += 1
            val_set.append( sample + [0] )
        # print( neg_idx )
        # input('>')
        
    if len( hist ) == 1:
        hist_item = hist
        sample = [ uid, hist[0], hist_item, len( hist_item ) ]
        last_col = 'label'
        val_set.append( sample +[1] )
        
print( val_set[:5] )
df_val = pd.DataFrame( val_set, columns = [ 'user_id', 'course_id', "hist_course_id", 'histlen_course_id', 'label' ] )
print( df_val[:5] )
print( max( df_val['histlen_course_id'] ) )

x_val = gen_model_input( df_val, user_df, 'user_id', course_df, 'course_id', seq_max_len = 50 )
y_val = x_val["label"]

print({k: v[:3] for k, v in x_val.items()})
print( y_val[:3] )
#input('>')
val_user = x_val 

from torch_rechub.basic.features import SparseFeature, SequenceFeature
user_features = [
    SparseFeature( 'user_id', vocab_size = user_df['user_id'].max() + 1, embed_dim = 16 ),
    SparseFeature( 'gender', vocab_size = user_df['gender'].max() + 1, embed_dim = 16 ),
    SparseFeature( 'occupation_titles', vocab_size = user_df['occupation_titles'].max() + 1, embed_dim = 16 ),
    SparseFeature( 'recreation_names', vocab_size = user_df['recreation_names'].max() + 1, embed_dim = 16 ),
    SparseFeature( 'groups_x', vocab_size = user_df['groups_x'].max() + 1, embed_dim = 16 ),
    SparseFeature( 'subgroups', vocab_size = user_df['subgroups'].max() + 1, embed_dim = 16 )

]
user_features += [
    SequenceFeature("hist_course_id",
                    vocab_size = course_df["course_id"].max() + 1,
                    embed_dim = 16,
                    pooling = "mean",
                    shared_with = "course_id")
]

course_features = [
    SparseFeature( 'course_id', vocab_size = course_df['course_id'].max() + 1, embed_dim = 16 ),
    SparseFeature( 'course_price', vocab_size = course_df['course_price'].max() + 1, embed_dim = 16 ),
    SparseFeature( 'teacher_id', vocab_size = course_df['teacher_id'].max() + 1, embed_dim = 16 ),
    SparseFeature( 'groups_y', vocab_size = course_df['groups_y'].max() + 1, embed_dim = 16, shared_with = 'groups_x' ),
    SparseFeature( 'sub_groups', vocab_size = course_df['sub_groups'].max() + 1, embed_dim = 16, shared_with = 'subgroups' ),
    SparseFeature( 'topics', vocab_size = course_df['topics'].max() + 1, embed_dim = 16 ),
    SparseFeature( 'course_published_at_local', vocab_size = course_df['course_published_at_local'].max() + 1, embed_dim = 16 ),
    SparseFeature( 'chapter_cnt', vocab_size = course_df['chapter_cnt'].max() + 1, embed_dim = 16 ),
    SparseFeature( 'unit_cnt', vocab_size = course_df['unit_cnt'].max() + 1, embed_dim = 16 ),
    SparseFeature( 'assignment_cnt', vocab_size = course_df['assignment_cnt'].max() + 1, embed_dim = 16 ),
    SparseFeature( 'total_sec', vocab_size = course_df['total_sec'].max() + 1, embed_dim = 16 )
]
print( user_features )
print( course_features )
#input('>')

from torch_rechub.utils.data import df_to_dict
all_item = df_to_dict( course_df )
print({k: v[:3] for k, v in all_item.items()})

from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer
from torch_rechub.utils.data import MatchDataGenerator

dg = MatchDataGenerator( x = x_train, y = y_train )
train_dl, val_dl,  item_dl = dg.generate_dataloader( x_val, all_item, batch_size = 256 )

model = DSSM( user_features,
              course_features,
              temperature = 0.02,
              user_params = {
                                "dims": [256, 128, 64],
                                "activation": 'prelu',  # important!!
              },
              item_params = {
                                "dims": [256, 128, 64],
                                "activation": 'prelu',  # important!!
              } )

trainer = MatchTrainer( model,
                        mode=0,  # 同上面的mode，需保持一致
                        optimizer_params = { "lr": 1e-4,
                                             "weight_decay": 1e-6 },
                        n_epoch = 10,
                        device = 'cuda',
                        model_path = './' )
trainer.fit( train_dl )

import collections
import numpy as np
import pandas as pd
from torch_rechub.utils.match import Annoy
from torch_rechub.basic.metric import topk_metrics
'''
def match_evaluation(user_embedding, item_embedding, test_user, all_item, user_col='user_id', item_col='course_id', topk=10):
    print("evaluate embedding matching on test data")
    annoy = Annoy(n_trees=10)
    annoy.fit(item_embedding)

    #for each user of test dataset, get ann search topk result
    print("matching for topk")
    
    #with open('../cache/vocab/user.json') as user_file:
    #    user_map= json.load(user_file)
    #with open('../cache/vocab/course.json') as item_file:
    #    item_map= json.load(item_file)
    match_res = collections.defaultdict(dict)  # user id -> predicted item ids
    
    for user_id, user_emb in zip(test_user[user_col], user_embedding):
        items_idx, items_scores = annoy.query(v=user_emb, n=topk)  #the index of topk match items
        #print(items_idx)
        match_res[user_id] = items_idx
        print( match_res[user_id])
        
    #get ground truth
    print("generate ground truth")
    data = pd.DataFrame({user_col: test_user[user_col], item_col: test_user[item_col]})
    #data[user_col] = data[user_col].map(user_map)
    #data[item_col] = data[item_col].map(item_map)
    user_pos_item = data.groupby(user_col).agg(list).reset_index()
    ground_truth = dict(zip(user_pos_item[user_col], user_pos_item[item_col]))  # user id -> ground truth

    #print((ground_truth))
    #print((match_res))
    print("compute topk metrics")
    out = topk_metrics(y_true=ground_truth, y_pred=match_res, topKs=[topk])
    return out

user_embedding = trainer.inference_embedding(model=model, mode="user", data_loader=val_dl, model_path='./')
item_embedding = trainer.inference_embedding(model=model, mode="item", data_loader=item_dl, model_path='./')


metrics = match_evaluation(user_embedding, item_embedding, val_user, all_item, topk=10)
print(metrics['Precision'])
print(metrics['Recall'])
print(metrics['Hit'])
'''

import csv
def Test(user_embedding, item_embedding, test_user, all_item, user_col='user_id', item_col='course_id', topk=10):
    print("evaluate embedding matching on test data")
    annoy = Annoy(n_trees=10)
    annoy.fit(item_embedding)
    #for each user of test dataset, get ann search topk result
    print("matching for topk")
    
    with open('../cache/vocab/user.json') as user_file:
        user_map= json.load(user_file)
    with open('../cache/vocab/course.json') as item_file:
        item_map= json.load(item_file)
    match_res = collections.defaultdict(dict)  # user id -> predicted item ids
    
    f = open('answer.csv' , 'w')
    writer = csv.writer(f)
    writer.writerow(['user_id', 'course_id'])
    
    for user_id, user_emb in zip(test_user[user_col], user_embedding):
        items_idx, items_scores = annoy.query(v=user_emb, n=topk)  #the index of topk match items
        print(items_idx)
        #match_res[user_id] = items_idx
        user_key_list = list(user_map.keys())
        user_val_list = list(user_map.values())
        position = user_val_list.index(user_id )
        ori_userid = user_key_list[position]
        ori_item = []
        for i in items_idx:
            item_key_list = list(item_map.keys())
            item_val_list = list(item_map.values())
            position = item_val_list.index(i )
            ori_item.append(item_key_list[position])
        
        x = ' '.join(ori_item)
        ret = []
        ret.append(ori_userid)
        ret.append(x)
        
        #print( match_res[user_id])
        writer.writerow(ret)
        


user_embedding = trainer.inference_embedding(model=model, mode="user", data_loader=val_dl, model_path='./')
item_embedding = trainer.inference_embedding(model=model, mode="item", data_loader=item_dl, model_path='./')
metrics = Test(user_embedding, item_embedding, val_user, all_item, topk=50)
print(metrics['Precision'])
print(metrics['Recall'])
print(metrics['Hit'])

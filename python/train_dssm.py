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
import argparse
from tqdm.auto import tqdm

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

def prepare_usr_df( usr_file, vocab_path ):
    print('Preparing user dataframe...')
    # usr data
    user_data = datasets.Dataset.from_csv(usr_file)
    user_p = BasicUserPreprocessor(vocab_path, column_names=user_data.column_names)

    batch_size = 32
    user_profile = prepare_user_datasets( user_data, user_p, batch_size, False )

    useless = [
        'interests',
    ]
    user_profile = user_profile.remove_columns(useless)

    # turn into df
    user_df = pd.DataFrame( user_profile, columns = user_profile.column_names )
    user_df.rename( columns = { 'groups': 'groups_user', 'subgroups': 'sub_groups_user' }, inplace = True )

    # padding sequence to the same length
    need_padding = [
        'occupation_titles',
        'recreation_names',
        'groups_user',
        'sub_groups_user'
    ]
    for col in need_padding:
        user_df[col] = pad_sequences( user_df[col], maxlen = 50, value = 0, padding = 'pre', truncating = 'pre' ).tolist()

    print('Done')

    return user_df, user_p

def prepare_course_df( course_file, vocab_path ):
    print('Preparing course dataframe...')
    # course data
    course_data = datasets.Dataset.from_csv(course_file)
    item_feat = course_item_features("../data/course_chapter_items.csv")
    course_p = BasicCoursePreprocessor(vocab_path,
                                       pretrained_name = "bert-base-multilingual-cased",
                                       column_names = course_data.column_names)

    batch_size = 32
    course_profile = prepare_course_datasets( course_data, item_feat, course_p, batch_size, False )
    useless =  [
        'course_name',
        'teacher_intro',
        'description',
        'will_learn',
        'required_tools',
        'recommended_background',
        'unit_cnt',
        'assignment_cnt',
        'chapter_cnt',
        'target_group'
    ]
    course_profile = course_profile.remove_columns(useless)

    # turn into df
    course_df = pd.DataFrame( course_profile, columns = course_profile.column_names )
    course_df = course_df.sort_values('course_id')
    course_df['course_id'] = ( np.array( course_df['course_id'] ) - 4 ).tolist()

    def normalize( df, cols ):
        for col in cols:
            temp = np.array( df[col], dtype = float )
            temp = ( temp - min( temp ) ) / ( max( temp ) - min( temp ) )
            df[col] = temp.tolist()
        return df

    # normalize dense features
    course_df = normalize( course_df, [ 'course_price', 'course_published_at_local', 'total_sec' ] )

    # padding sequence to the same length
    need_padding = [ 'groups', 'sub_groups', 'topics' ]
    for col in need_padding:
        course_df[col] = pad_sequences( course_df[col], maxlen = 50, value = 0, padding = 'pre', truncating = 'pre' ).tolist()

    print('Done')
    return course_df, course_p


def prepare_train_dataset( train_file, user_df, course_df, user_p, course_p ):
    print('Preparing train dataset...')
    # train data
    train_data = pd.read_csv(train_file)
    train_data['user_id'] = user_p.encode_user_id( train_data['user_id'] )

    encoded_course_id = []
    for hist in train_data['course_id']:
        encoded_course_id.append( ( np.array( course_p.encode_course_id( hist.split(' ') ) ) - 4 ).tolist() )

    train_data['course_id'] = encoded_course_id

    # negative sampling
    print('Negative Sampling')
    neg_rat = 3
    items_cnt = Counter( [ course for hist in train_data['course_id'] for course in hist ] )
    items_cnt_order = OrderedDict(sorted((items_cnt.items()), key=lambda x: x[1], reverse=True))
    neg_list = negative_sample(items_cnt_order, ratio=139608 * neg_rat, method_id=1)

    train_set = []
    neg_idx = 0
    for i in tqdm(range( len( train_data ) )):
        uid = train_data['user_id'][i]
        pos_list = train_data['course_id'][i]
        for j in range( 1, len( pos_list ) ):
            hist_item = pos_list[:j]
            sample = [ uid, pos_list[j], hist_item, len( hist_item ) ]
            if j != len( pos_list ) - 1:
                last_col = 'label'
                train_set.append( sample + [1] )
                for _ in range( neg_rat ):
                    sample[1] = neg_list[neg_idx]
                    neg_idx += 1
                    train_set.append( sample + [0] )

    random.shuffle(train_set)
    df_train = pd.DataFrame( train_set, columns = [ 'user_id', 'course_id', "hist_course_id", 'histlen_course_id' ] + [ last_col ] )

    x_train = gen_model_input( df_train, user_df, 'user_id', course_df, 'course_id', seq_max_len = 50 )
    y_train = x_train["label"]

    print('Done')
    return x_train, y_train

def prepare_val_dataset( val_file, user_df, course_df, user_p, course_p ):
    print('Preparing validation dataset...')
    val_data = pd.read_csv(val_file)
    val_data['user_id'] = user_p.encode_user_id( val_data['user_id'] )

    encoded_course_id = []
    for hist in val_data['course_id']:
        encoded_course_id.append( ( np.array( course_p.encode_course_id( hist.split(' ') ) ) - 4 ).tolist() )

    val_data['course_id'] = encoded_course_id

    val_set = []
    for i in tqdm( range( len( val_data ) ) ):
        uid = val_data['user_id'][i]
        hist = val_data['course_id'][i]
        sample = [ uid, hist[len(hist) - 1], hist, len( hist ) ]
        last_col = 'label'
        val_set.append( sample + [1] )

    df_val = pd.DataFrame( val_set, columns = [ 'user_id', 'course_id', "hist_course_id", 'histlen_course_id' ] + [ last_col ] )

    x_val = gen_model_input( df_val, user_df, 'user_id', course_df, 'course_id', seq_max_len = 50 )
    y_val = x_val["label"]

    print('Done')
    return x_val, val_data['course_id']

def prepare_test_dataset( test_file, user_df, course_df, user_p, course_p ):
    print('Preparing test dataset...')
    test_data = pd.read_csv(test_file)
    test_data['user_id'] = user_p.encode_user_id( test_data['user_id'] )

    encoded_course_id = []
    for hist in test_data['course_id']:
        encoded_course_id.append( ( np.array( course_p.encode_course_id( hist.split(' ') ) ) - 4 ).tolist() )

    test_data['course_id'] = encoded_course_id

    test_set = []
    for i in tqdm( range( len( test_data ) ) ):
        uid = test_data['user_id'][i]
        hist = test_data['course_id'][i]
        sample = [ uid, hist[len(hist) - 1], hist, len( hist ) ]
        last_col = 'label'
        test_set.append( sample + [1] )

    df_test = pd.DataFrame( test_set, columns = [ 'user_id', 'course_id', "hist_course_id", 'histlen_course_id' ] + [ last_col ] )

    x_test = gen_model_input( df_test, user_df, 'user_id', course_df, 'course_id', seq_max_len = 50 )
    y_test = x_test["label"]

    print('Done')
    return x_test, y_test

def create_model( embed_size, temp, dropout, lr, weight_decay, epoch, model_path ):
    print('Creating user & course features and building model & trainer...', flush=True)
    user_features = [
        SparseFeature( 'user_id', vocab_size = 130566, embed_dim = embed_size ),
        SparseFeature( 'gender', vocab_size = 7, embed_dim = embed_size ),
    ]
    user_features += [
        SequenceFeature("hist_course_id",
                        vocab_size = 728,
                        embed_dim = embed_size,
                        pooling = "mean",
                        shared_with = "course_id"),
        SequenceFeature("groups_user",
                        vocab_size = 16,
                        embed_dim = embed_size,
                        pooling = "mean"),
        SequenceFeature("sub_groups_user",
                        vocab_size = 99,
                        embed_dim = embed_size,
                        pooling = "mean"),
        SequenceFeature("occupation_titles",
                        vocab_size = 24,
                        embed_dim = embed_size,
                        pooling = "mean"),
        SequenceFeature("recreation_names",
                        vocab_size = 35,
                        embed_dim = embed_size,
                        pooling = "mean"),
    ]

    course_features = [
        SparseFeature( 'course_id', 728, embed_dim = embed_size ),
        SparseFeature( 'teacher_id', 557, embed_dim = embed_size ),
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

    model = DSSM( user_features,
                  course_features,
                  temperature = temp,
                  user_params = {
                      "dims": [256, 128, 64],
                      "activation": 'prelu',
                      "dropout": dropout,
                  },
                  item_params = {
                      "dims": [256, 128, 64],
                      "activation": 'prelu',
                      "dropout": dropout,
                  } )
    print( model )
    trainer = MatchTrainer(model,
                           mode=0,
                           optimizer_params = {
                               "lr": lr,
                               "weight_decay": weight_decay
                           },
                           n_epoch = epoch,
                           device = 'cuda',
                           model_path = model_path )
    print('Done')
    return trainer, model

def val(user_embedding, item_embedding, test_user, all_item, label, user_col='user_id', item_col='course_id', topk=10):
    print("Evaluating...")
    annoy = Annoy(n_trees=topk)
    annoy.fit(item_embedding)

    print("matching for topk")
    preds = []
    for user_id, user_emb in tqdm(zip( test_user[user_col], user_embedding )):
        items_idx, items_scores = annoy.query(v=user_emb, n=topk)  #the index of topk match items
        preds.append( items_idx )

    print("Computing MAP@K metrics")
    score = mapk( label, preds, k = topk )
    print( 'MAP@50:', score )

def test(user_embedding, item_embedding, test_user, all_item, output_file, user_col='user_id', item_col='course_id', topk=10):
    print("Testing...")
    annoy = Annoy(n_trees=topk)
    annoy.fit(item_embedding)
    #for each user of test dataset, get ann search topk result
    print("matching for topk")

    with open('../cache/vocab/user.json') as user_file:
        user_map= json.load(user_file)
    with open('../cache/vocab/course.json') as item_file:
        item_map= json.load(item_file)

    f = open(output_file, 'w')
    writer = csv.writer(f)
    writer.writerow(['user_id', 'course_id'])

    user_key_list = list(user_map.keys())
    user_val_list = list(user_map.values())
    item_key_list = list(item_map.keys())
    item_val_list = list(item_map.values())
    for user_id, user_emb in tqdm(zip(test_user[user_col], user_embedding)):
        items_idx, items_scores = annoy.query(v=user_emb, n=topk)  #the index of topk match items
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

    print(f"Save predicions to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description = "Generates personalized recommendations for each user",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type = str,
        default = "predictions.csv",
        dest = "output_file",
        help = "output file name",
    )
    parser.add_argument(
        "--model_path",
        type = str,
        default = "models/",
        dest = "model_path",
        help = "model saved directory",
    )
    parser.add_argument(
        "--user_file",
        type = str,
        default = "../data/users.csv",
        dest = "user_file",
        help = "user data path",
    )
    parser.add_argument(
        "--course_file",
        type = str,
        default = "../data/courses.csv",
        dest = "course_file",
        help = "course data path",
    )
    parser.add_argument(
        "--vocab_path",
        type = str,
        default = "../cache/vocab",
        dest = "vocab_path"
    )
    parser.add_argument(
        "--train_file",
        type = str,
        default = "../data/train.csv",
        dest = "train_file",
        help = "train data path",
    )
    parser.add_argument(
        "--val_file",
        type = str,
        default = "../data/val_seen.csv",
        dest = "val_file",
        help = "val data path",
    )
    parser.add_argument(
        "--test_file",
        type = str,
        default = "../data/test_seen.csv",
        dest = "test_file",
        help = "test data path",
    )
    parser.add_argument(
        "--test", action = "store_true"
    )
    parser.add_argument(
        "--embed_size",
        type = int,
        default = 16,
        dest = "embed_size",
        help = "user & course's embedding size",
    )
    parser.add_argument(
        "--temp",
        type = float,
        default = 1,
        dest = "temp",
        help = "temperature",
    )
    parser.add_argument(
        "--dropout",
        type = float,
        default = 0,
        help = "doupout rate",
    )
    parser.add_argument(
        "--lr",
        type = float,
        default = 1e-4,
        dest = "lr",
        help = "learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type = float,
        default = 1e-6,
        help = "weight decay",
    )
    parser.add_argument(
        "--epoch",
        type = int,
        default = 10,
        help = "number of training epoch",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    #####################
    ### user & course data
    #####################
    user_df, user_p = prepare_usr_df( args.user_file, args.vocab_path )
    course_df, course_p = prepare_course_df( args.course_file, args.vocab_path )

    #####################
    ### train dataset & trainer
    #####################
    x_train, y_train = prepare_train_dataset( args.train_file, user_df, course_df, user_p, course_p )
    trainer, model = create_model( args.embed_size, args.temp, args.dropout, args.lr, args.weight_decay, args.epoch, args.model_path )

    #####################
    ### val & test dataset
    #####################
    x_test, y_test = prepare_test_dataset( args.test_file, user_df, course_df, user_p, course_p )
    x_val, y_val = prepare_val_dataset( args.val_file, user_df, course_df, user_p, course_p )

    #####################
    ### dataloaders
    #####################
    all_item = df_to_dict( course_df )

    dg = MatchDataGenerator( x = x_train, y = y_train )
    train_dl, val_dl,  item_dl = dg.generate_dataloader( x_val, all_item, batch_size = 4096 )
    _, test_dl,  item_dl = dg.generate_dataloader( x_test, all_item, batch_size = 4096 )

    #####################
    ### training
    #####################
    if not args.test:
        print('Training...')
        trainer.fit( train_dl )

    #####################
    ### validation
    #####################
    user_embedding = trainer.inference_embedding(model=model, mode="user", data_loader=val_dl, model_path=args.model_path)
    item_embedding = trainer.inference_embedding(model=model, mode="item", data_loader=item_dl, model_path=args.model_path)
    val(user_embedding, item_embedding, x_val, all_item, y_val, topk=50)

    #####################
    ### testing
    #####################
    if args.test:
        user_embedding = trainer.inference_embedding(model=model, mode="user", data_loader=test_dl, model_path=args.model_path)
        item_embedding = trainer.inference_embedding(model=model, mode="item", data_loader=item_dl, model_path=args.model_path)
        test(user_embedding, item_embedding, x_test, all_item, topk=50, output_file=args.output_file)

if __name__ == '__main__':
    seed = 9527
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed( seed )
    np.random.seed( seed )
    random.seed( seed )
    main()

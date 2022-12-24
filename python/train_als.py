import torch
import pandas as pd
import numpy as np
import os
import csv
import json
from typing import *
from pathlib import Path
import datasets
import argparse
import logging
from scipy import sparse
from tqdm import tqdm

from utils.user_preprocessor import BasicUserPreprocessor, prepare_user_datasets
from utils.course_preprocessor import BasicCoursePreprocessor, prepare_course_datasets, course_item_features
from utils.average_precision import *

from implicit.als import AlternatingLeastSquares
from implicit.approximate_als import (
    AnnoyAlternatingLeastSquares,
    FaissAlternatingLeastSquares,
    NMSLibAlternatingLeastSquares,
)
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization
from implicit.nearest_neighbours import (
    BM25Recommender,
    CosineRecommender,
    TFIDFRecommender,
    bm25_weight,
)

os.environ["OMP_NUM_THREADS"] = "1"
# pd.set_option( 'display.max_rows', 500 )
# pd.set_option( 'display.max_columns', 500 )
# pd.set_option( 'display.width', 1000 )
np.set_printoptions( threshold = np.inf )
torch.manual_seed(9527)

MODELS = {
    "als": AlternatingLeastSquares,
    "nmslib_als": NMSLibAlternatingLeastSquares,
    "annoy_als": AnnoyAlternatingLeastSquares,
    "bpr": BayesianPersonalizedRanking,
    "lmf": LogisticMatrixFactorization,
}

def get_model( model_name ):
    print(f"getting model {model_name}")
    model_class = MODELS.get( model_name )
    if not model_class:
        raise ValueError(f"Unknown Model '{model_name}'")

    # some default params
    if model_name.endswith("als"):
        params = { "factors": 76, "regularization": 2, "alpha": 1.5,"dtype": np.float32, "iterations": 200, "calculate_training_loss": True }
    elif model_name == "bm25":
        params = { "K1": 100, "B": 0.5 }
    elif model_name == "bpr":
        params = { "factors": 64, "iterations": 200 }
    elif model_name == "lmf":
        params = { "factors": 256, "iterations": 500, "regularization": 2.5, "learning_rate": 1.2, "random_state": 9527, "neg_prop": 50 }
    else:
        params = {}

    return model_class(**params)

def prepare_training_data( train_file, vocab_file ):
    print('**** Preparing training data... ****')
    # load training data
    # encode user id
    train_data = pd.read_csv( train_file )
    user_p = BasicUserPreprocessor( vocab_file, column_names = [ 'user_id' ] )
    train_data['user_id'] = user_p.encode_user_id( train_data['user_id'] )

    # encode course id
    course_p = BasicCoursePreprocessor( vocab_file, pretrained_name = "bert-base-multilingual-cased", column_names = [ 'course_id' ] )
    encoded_course_id = []
    for hist in train_data['course_id']:
        encoded_course_id.append( ( np.array( course_p.encode_course_id( hist.split(' ') ) ) - 4 ).tolist() )

    # print( min( encoded_course_id ) )
    # input('>')
    train_data['course_id'] = encoded_course_id

    # user-course matrix
    print('**** Constructing user-course matrix... ****')
    num_users = 130565
    num_courses = 731 - 4
    uc_matrix = np.zeros( shape = ( num_courses, num_users ), dtype = int )

    for i, data in tqdm( train_data.iterrows() ):
        for course in data['course_id']:
             uc_matrix[course][data['user_id']] = 1

    print('**** Done... ****')
    return uc_matrix

def prepare_eval_data( eval_file, vocab_file ):
    print('**** Preparing validation data ****')
    # encode user id
    eval_data = pd.read_csv( eval_file )
    user_p = BasicUserPreprocessor( vocab_file, column_names = [ 'user_id' ] )
    eval_data['user_id'] = user_p.encode_user_id( eval_data['user_id'] )

    # encode course id
    course_p = BasicCoursePreprocessor( vocab_file, pretrained_name = "bert-base-multilingual-cased", column_names = [ 'course_id' ] )
    encoded_course_id = []
    for hist in eval_data['course_id']:
        encoded_course_id.append( ( np.array( course_p.encode_course_id( hist.split(' ') ) ) - 4 ).tolist() )

    eval_data['course_id'] = encoded_course_id

    print('**** Done... ****')
    return eval_data

def prepare_test_data( test_file, vocab_file ):
    print('**** Preparing testing data ****')
    # encode user id
    test_data = pd.read_csv( test_file )
    user_p = BasicUserPreprocessor( vocab_file, column_names = [ 'user_id' ] )
    test_data['user_id'] = user_p.encode_user_id( test_data['user_id'] )

    print('**** Done... ****')
    return test_data

def train( model_path, model_name = "als" ):
    train_data = prepare_training_data( '../data/train.csv', '../cache/vocab' )
    uc_matrix = sparse.csr_matrix( train_data )

    model = get_model( model_name )
    # if we're training an ALS based model, weight input by bm25
    if model_name.endswith("als"):
        # lets weight these models by bm25weight.
        print("weighting matrix by bm25_weight")
        uc_matrix = bm25_weight( uc_matrix, K1 = 250, B = 0.7 )

        # also disable building approximate recommend index
        model.approximate_similar_items = False

    matrix = uc_matrix.T.tocsr()

    print('**** Starting training... ****')
    model.fit( matrix )

    # data = {
    # 'model.item_factors': model.item_factors,
    # 'model.user_factors': model.user_factors,
    # }
    # np.savez( model_path, **data )

    return model, matrix

def val( model, matrix ):
    # load validation data
    eval_data = prepare_eval_data( '../data/val_seen.csv', '../cache/vocab' )
    # load matrix
    # train_data = prepare_training_data( '../data/train.csv', '../cache/vocab' )
    # uc_matrix = bm25_weight( train_data, K1 = 100, B = 0.8 )
    # matrix = uc_matrix.T.tocsr()

    # load model
    # data = np.load( 'models/user_course.npz', allow_pickle = True )
    # model = AlternatingLeastSquares( factors = data['model.item_factors'].shape[1] )
    # model.item_factors = data['model.item_factors']
    # model.user_factors = data['model.user_factors']
    # model._YtY = model.item_factors.T.dot( model.item_factors )

    print('**** Starting prediction... ****')
    preds = []
    for user in tqdm( eval_data['user_id'] ):
        ids, _ = model.recommend( user, matrix[user], N = 50, filter_already_liked_items = True )
        # print( _ )
        preds.append( ids )

    score = mapk( eval_data['course_id'], preds, k = 50 )
    print( 'MAP@50:', score )

def test( model, matrix ):
    test_data = prepare_test_data( '../data/test_seen.csv', '../cache/vocab' )

    f = open('answer.csv' , 'w')
    writer = csv.writer(f)
    writer.writerow(['user_id', 'course_id'])

    with open('../cache/vocab/user.json') as user_file:
        user_map = json.load(user_file)
    with open('../cache/vocab/course.json') as item_file:
        item_map = json.load(item_file)

    a = 0
    item_key_list = list( item_map.keys() )
    item_val_list = list( item_map.values() )
    user_key_list = list( user_map.keys() )
    user_val_list = list( user_map.values() )
    for user in tqdm( test_data['user_id'] ):
        ori_cor = []
        ids, scores = model.recommend( user , matrix[user], N = 50, filter_already_liked_items = True )
        for j in ids:
            position = item_val_list.index( j + 4 )
            ori_cor.append( item_key_list[position] )

        position = user_val_list.index( user )
        ori_userid = user_key_list[position]
        recom = ' '.join(ori_cor)
        ret = []
        ret.append(ori_userid)
        ret.append(recom)
        writer.writerow(ret)
        a += 1
        # print(a)
    print(a)

def main():
    parser = argparse.ArgumentParser(
        description = "Generates personalized recommendations for each user",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type = str,
        default = "predictions.csv",
        dest = "outputfile",
        help = "output file name",
    )
    parser.add_argument(
        "--model",
        type = str,
        default = "als",
        dest = "model",
        help = f"model to calculate ({'/'.join(MODELS.keys())})",
    )
    parser.add_argument(
        "--model_path",
        type = str,
        default = "models/user-course.npz",
        dest = "model_path"
    )
    parser.add_argument(
        "--param", action = "append", help = "Parameters to pass to the model, formatted as 'KEY=VALUE"
    )
    parser.add_argument(
        "--test", action = "store_true"
    )

    args = parser.parse_args()

    logging.basicConfig( level = logging.DEBUG )

    print('**** Training... ****')
    model, matrix = train( args.model_path, args.model )
    print('**** Evaluating... ****')
    val( model, matrix )
    if args.test:
        print('**** Testing... ****')
        test( model, matrix )

if __name__ == '__main__':
    main()


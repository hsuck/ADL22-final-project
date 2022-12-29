import pandas as pd
import numpy as np
from numpy.random import choice
import torch
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from pathlib import Path
from typing import List

from utils.evaluate import evaluate_map

from matplotlib import pyplot as plt

from scipy.stats import norm
from functools import partial

data_dir = Path("../data")


def initialize():
    train_df = pd.read_csv( str(data_dir/'train.csv') ).fillna('')
    course_df = pd.read_csv( str(data_dir/"courses.csv") ).fillna('')
    user_plog = {
        uid: cid_list.split(' ') for uid, cid_list in zip(train_df['user_id'], train_df['course_id'])
    }
    course_feat = {
        feat['course_id']: feat.drop('course_id').to_dict() for feat in course_df.iloc
    }

    subgroup_df = pd.read_csv( str(data_dir/'subgroups.csv') )
    sg2id = {
        name: id for name, id in zip( subgroup_df['subgroup_name'], subgroup_df['subgroup_id'])
    }
    sg2id[''] = -1
    
    return user_plog, course_feat, sg2id

user_plog, course_feat, sg2id = initialize()


#######################
### Utils 
#######################

def normalize( x ):
    s = np.sum(x)
    return np.array(x) / s

def softmax(x):
    return torch.softmax( input=x, dim=0)

def to_submission_str( data: List, should_sort: bool ):
    if should_sort:
        data = sorted(data)
    return " ".join( [str(x) for x in data] )

def sample_over_prob(candidate: List, prob, num_of_sample: int, should_sort: bool) -> pd.DataFrame:
    return [
        to_submission_str(choice(candidate, 50, replace=False, p=prob), should_sort) for _ in range(num_of_sample)
    ]

def get_purchase_count() -> Counter:
    course_cnt = [ x for plog in user_plog.values() for x in plog ]
    course_cnt = Counter(course_cnt)
    return course_cnt


#######################
### Price Distribution
#######################
def get_price_distribution():
    pcnt = get_purchase_count()

    def def_value(): return 0
    res = defaultdict(def_value)
    for cid, cnt in pcnt.items():
        price = course_feat[cid]['course_price']
        res[price] += cnt

    data = []
    for key, val in res.items():
        if key == 0: continue
        data += [key] * val
    mu, std = norm.fit(data)

    def pfunc(x, mu, std):
        return ( norm.pdf(x, mu, std)*1.5 + norm.pdf(x, 0, std/15) ) /2

    prob_func = partial( pfunc, mu=mu, std=std )
    return Counter(res), prob_func

def draw_price_distribution():
    price_cnt, prob_func = get_price_distribution()

    data = []
    for key, val in price_cnt.items():
        data += [key] * val

    plt.hist(data, bins=[x for x in range(0, max(data), 100)], density=True, alpha=0.6, color='g')
    plt.savefig("hist.png")

    _, xmax = plt.xlim()
    x = np.linspace(0, xmax, 2000)
    p = [ prob_func(i) for i in x ]
    plt.plot(x, p, 'k', linewidth=2)
    plt.title("P( course is be bought | price)")

    plt.savefig("price_distribution.png")


#######################
### Publish Time Distribution
#######################
def get_pubtime_distribution():
    pcnt = get_purchase_count()

    def def_value(): return 0
    res = defaultdict(def_value)

    for cid, cnt in pcnt.items():
        time = course_feat[cid]['course_published_at_local'][:7]
        res[time] += cnt

    year_prob = np.zeros( (7, 12) )
    for year in range(2015, 2022):
        for month in range(1, 13):
            year_prob[year-2015][month-1] = res[ f"{year}-{month}" ]
    
        s = np.sum(year_prob[year-2015])
        year_prob[year-2015] = year_prob[year-2015] / s
    
    month_prob_arr = np.mean( year_prob[1:-1,:], axis=0)
    month_prob_arr[ month_prob_arr == 0 ] = np.min(month_prob_arr) / 5
    def pfunc(year, month, month_prob_arr):
        """
            month in [1, 12]
        """
        year_weight = np.max( [ 2*np.exp(year-2015), 1e-9])
        return month_prob_arr[ month - 1 ] * year_weight

    prob_func = partial(
        pfunc,
        month_prob_arr = month_prob_arr,
    )

    return Counter(res), prob_func

def draw_pubtime_distribution():
    pubtime_cnt, prob_func = get_pubtime_distribution()

    pubtime = []
    cnt = []
    prob = []
    for year in range(2015, 2022):
        for month in range(1, 13):
            t = f"{year}-{month}"
            pubtime.append(t)
            cnt.append(pubtime_cnt[t])
            prob.append( prob_func(year, month) ) 
        
        print(pubtime[-12:])
        print(cnt[-12:])
    prob = np.array(prob)

    total = np.sum(cnt)
    cnt = np.array(cnt) / total

    plt.figure(figsize=(24,8), facecolor='white')
    plt.xticks(rotation=90, ha='right')
    plt.bar(
        x = pubtime,
        height = cnt,
        label = 'train'
    )

    prob = prob / np.sum(prob)

    print(prob)
    plt.bar(
        x = pubtime,
        height = prob,
        alpha=0.5,
        label = 'estimate'
    )

    ymin, ymax = plt.ylim()
    date_sep = list( range(11, 11+7*12, 12) )
    plt.vlines(
        x = date_sep,
        ymin = [ymin] * len(date_sep),
        ymax = [ymax] * len(date_sep),
        linestyles=['dotted'] * len(date_sep)
    )

    plt.legend()
    plt.title("P( course is be bought | Publis Time)")

    plt.savefig("pubtime_distribution.png")



#######################
### Joint Price & Pub-Time Distribution
#######################
def get_joint_price_pubtime_distribution():
    pcnt = get_purchase_count()

    def def_value(): return 0
    res = defaultdict(def_value)

    for cid, cnt in pcnt.items():
        price = course_feat[cid]['course_price']
        time = course_feat[cid]['course_published_at_local'][:7]
        year, month = int(time[:4]), int(time[-2:])
        res[(price,year,month)] += cnt

    return Counter(res)

def draw_joint_price_pubtime_distribution():
    pp_cnt = get_joint_price_pubtime_distribution()
    total_cnt = np.sum( list(pp_cnt.values()) )

    print(pp_cnt)

#######################
### Subgroup Distribution
#######################
def get_subgroup_distribution():
    pcnt = get_purchase_count()

    def def_value(): return 0
    res = defaultdict(def_value)


    for cid, cnt in pcnt.items():
        sgs = [ sg2id[sg] for sg in course_feat[cid]['sub_groups'].split(',') ]
        for sg in sgs:
            res[sg] += cnt
        
    total = np.sum( list(res.values()) )
    category_prob = {
        key: val / total for key, val in res.items()
    }
    def pfunc( subgroups: str, category_prob: Dict[int, float] ):
        # subgroups = "生活,語言,..."
        sgs = [ sg2id[sg] for sg in subgroups.split(',')]
        probs = [ category_prob[i] for i in sgs]
        return np.mean(probs)

    
    prob_func = partial(
        pfunc,
        category_prob=category_prob
    )
    return Counter(res), prob_func

def draw_joint_price_pubtime_distribution():
    pp_cnt = get_joint_price_pubtime_distribution()
    total_cnt = np.sum( list(pp_cnt.values()) )

    print(pp_cnt)

#######################
### Inference & Eval
#######################

def get_course_prob():
    _, price_pfunc = get_price_distribution()
    _, pub_pfunc = get_pubtime_distribution()
    _, sg_pfunc = get_subgroup_distribution()
    course_prob = {}


    for cid, feat in course_feat.items():
        try:
            price = feat['course_price']
            time = feat['course_published_at_local'][:7]
            y = int(time[:4])
            m = int(time[-2:])
            sg_str = feat['sub_groups']
            course_prob[cid] = price_pfunc( price ) * pub_pfunc(y, m) * sg_pfunc(sg_str)
        except Exception as e:
            print(e)
            course_prob[cid] = 0

    return course_prob


def generate_course_pred_csv( input_file_path: Path, course_list: List, prob, output_dir = Path("../output/sample") ):
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv( str(input_file_path) )
    col = 'course_id'
    should_sort = ( col == 'subgroup')

    if 'unseen' in input_file_path.name:
        user_num = len(df)
        pred = sample_over_prob( course_list, prob=prob, num_of_sample=user_num, should_sort=should_sort)
        df[col] = pred
    else:
        pred = []
        for uid in df['user_id']:
            uprob = np.array( prob )
            for cid in user_plog[uid]:
                idx = course_list.index(cid)
                uprob[idx] = 0 
            uprob = normalize(uprob)
            user_pred = sample_over_prob( course_list, prob=uprob, num_of_sample=1, should_sort=should_sort)
            pred += user_pred
        
        df[col] = pred

    output_csv = str(output_dir / f"{input_file_path.stem}_pred.csv")
    df.to_csv( output_csv, index=False )
    return output_csv



def uniform_sample():
    eval_files = ['val_seen.csv', 'val_unseen.csv', 'val_seen_group.csv', 'val_unseen_group.csv' ]
    test_files = ['test_seen.csv', 'test_unseen.csv', 'test_seen_group.csv', 'test_unseen_group.csv' ]

    course_list = pd.read_csv("../data/courses.csv")['course_id']
    topic_list = pd.read_csv("../data/subgroups.csv")['subgroup_id']

    output_dir = Path("../output/uniform_sample")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in eval_files:
        file_path = data_dir / name
        df = pd.read_csv( str(file_path) ).fillna('')
        if 'group' in name:
            pred = sample_over_prob(topic_list, prob=None, num_of_sample=len(df), should_sort=True)
            df['subgroup'] = pred
        else:
            pred = sample_over_prob(course_list, prob=None, num_of_sample=len(df), should_sort=False)
            df['course_id'] = pred

        output_path = output_dir / f"{file_path.stem}_random.csv"
        df.to_csv( str(output_path), index=False )

        print(f"{file_path.stem}:\n  mAP@50: {evaluate_map( str(file_path), str(output_path) )}")
    
    for name in test_files:
        file_path = data_dir / name
        df = pd.read_csv( str(file_path) ).fillna('')
        if 'group' in name:
            pred = sample_over_prob(topic_list, prob=None, num_of_sample=len(df), should_sort=True)
            df['subgroup'] = pred
        else:
            pred = sample_over_prob(course_list, prob=None, num_of_sample=len(df), should_sort=False)
            df['course_id'] = pred

        output_path = output_dir / f"{file_path.stem}_random.csv"
        df.to_csv( str(output_path), index=False )



if __name__ == "__main__":
    """                   val           /  test 
        course seen:    0.0055 - 0.0065 / 0.00611
        course unseen:  0.0055 - 0.0065 / 0.00577
        topic seen:     0.0780 - 0.0850 / 0.07525
        topic unseen:   0.0710 - 0.0750 / 0.08913
    """
    # draw_joint_price_pubtime_distribution()

    course2prob = get_course_prob()
    courses = list(course2prob.keys())
    prob = normalize(list(course2prob.values()))

    from utils.submission_format import course_to_topic_pred
    
    output_dir = Path("../output/month_price_sample")
    eval_files = ['val_seen.csv', 'val_unseen.csv' ]
    for name in eval_files:
        file_path = data_dir / name
        output_path = generate_course_pred_csv( file_path, courses, prob, output_dir )

        topic_actual = data_dir / f"{file_path.stem}_group.csv"
        topic_pred = output_dir / f"{file_path.stem}_group_pred.csv"
        course_to_topic_pred(
                str(output_path),
                "../data/courses.csv",
                "../data/subgroups.csv",
        ).to_csv( str(topic_pred), index=False )

        print(f"{file_path.stem}:")
        print(f"  course mAP@50: {evaluate_map( str(file_path), str(output_path) )}")
        print(f"  topic  mAP@50: {evaluate_map( str(topic_actual), str(topic_pred) )}")

    test_files = ['test_seen.csv', 'test_unseen.csv' ]
    for name in test_files:
        file_path = data_dir / name
        
        print(f"Inference {file_path.stem} Course / Topic Prediction ...")
        output_path = generate_course_pred_csv( file_path, courses, prob, output_dir )
        topic_pred = output_dir / f"{file_path.stem}_group_pred.csv"
        course_to_topic_pred(
                str(output_path),
                "../data/courses.csv",
                "../data/subgroups.csv",
        ).to_csv( str(topic_pred), index=False )
        print("Finish")

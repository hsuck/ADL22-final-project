import pandas as pd
import numpy as np

from pathlib import Path
from typing import List

from utils.evaluate import evaluate_map

def to_submission_str( data: List, should_sort: bool ):
    if should_sort:
        data = sorted(data)
    return " ".join( [str(x) for x in data] )

def random_ans(candidate: List, num: int, should_sort: bool) -> pd.DataFrame:
    return [
        to_submission_str(np.random.choice(candidate, 50), should_sort) for _ in range(num)
    ]


def main():
    data_dir = Path("../data")
    eval_files = ['val_seen.csv', 'val_unseen.csv', 'val_seen_group.csv', 'val_unseen_group.csv' ]
    test_files = ['test_seen.csv', 'test_unseen.csv', 'test_seen_group.csv', 'test_unseen_group.csv' ]

    course_list = pd.read_csv("../data/courses.csv")['course_id']
    topic_list = pd.read_csv("../data/subgroups.csv")['subgroup_id']

    output_dir = Path("../output")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in eval_files:
        file_path = data_dir / name
        df = pd.read_csv( str(file_path) ).fillna('')
        if 'group' in name:
            pred = random_ans(topic_list, len(df), True)
            df['subgroup'] = pred
        else:
            pred = random_ans(course_list, len(df), False)
            df['course_id'] = pred

        output_path = output_dir / f"{file_path.stem}_random.csv"
        df.to_csv( str(output_path), index=False )

        print(f"{file_path.stem}:\n  mAP@50: {evaluate_map( str(file_path), str(output_path) )}")
    
    for name in test_files:
        file_path = data_dir / name
        df = pd.read_csv( str(file_path) ).fillna('')
        if 'group' in name:
            pred = random_ans(topic_list, len(df), True)
            df['subgroup'] = pred
        else:
            pred = random_ans(course_list, len(df), False)
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
    main()
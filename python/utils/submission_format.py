import pandas as pd
import numpy as np
def course_to_topic_pred(
        user_course_pred_csv: str,
        course_csv: str,
        topic_csv: str
    ) -> pd.DataFrame:

    df_pred = pd.read_csv(user_course_pred_csv).fillna('')
    df_info = pd.read_csv(course_csv).fillna('')
    df_topic = pd.read_csv(topic_csv)

    t2id = {
        topic: id
        for id, topic
        in zip(df_topic['subgroup_id'], df_topic['subgroup_name'])
    }

    c2t = {
        cid: [ t2id[name] for name in topics.split(',') ] if topics != '' else []
        for cid, topics
        in zip(df_info['course_id'], df_info['sub_groups'])
    }

    
    df_pred['subgroup'] = [
        sorted( np.unique( [ tid for cid in cids.split(' ') for tid in c2t[cid]] ) )
        for cids in df_pred['course_id']
    ]
    df_pred['subgroup'] = [
        ' '.join( [ str(x) for x in topic_list ] )
        for topic_list in df_pred['subgroup']
    ]
    return df_pred.drop(columns=['course_id'])

if __name__ == "__main__":
    df = course_to_topic_pred(
        "../../data/val_seen.csv",
        "../../data/courses.csv",
        "../../data/subgroups.csv"
    )
    print(df)

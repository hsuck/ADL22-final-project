from utils.user_preprocessor import BasicUserPreprocessor
from utils.submission_format import course_to_topic_pred
from random_50 import get_subgroup_distribution
import numpy as np
import pandas as pd
from collections import OrderedDict, Counter

if __name__ == "__main__":
    # user_p = BasicUserPreprocessor("../cache/vocab", column_names=['user_id'])

    user_df = course_to_topic_pred(
        "../data/val_seen.csv",
        "../data/courses.csv",
        "../data/subgroups.csv"
    )
    # user_df['user_id'] = user_p.encode_user_id( user_df['user_id'] )
    print( user_df )

    df = course_to_topic_pred(
        "./predictions.csv",
        "../data/courses.csv",
        "../data/subgroups.csv"
    )
    # df['user_id'] = user_p.encode_user_id( df['user_id'] )
    print(df)

    _, sg_pfunc = get_subgroup_distribution()
    print( _, sg_pfunc )

    df = pd.merge( df, user_df, how = 'left', on = 'user_id' )
    df = df.fillna('')
    print( df )
    sg = []
    for i, data in df.iterrows():
        # print( data['subgroup_x'], data['subgroup_y'], sep = '\n' )
        # input('>')
        subgroup_set_1 = data['subgroup_x'].split(' ')
        # subgroup_set_1 = set( [ int(s) for s in subgroup_set_1 ] )
        if data['subgroup_y'] == '':
            # subgroup_set_2 = set()
            subgroup_set_2 = []
        else:
            subgroup_set_2 = data['subgroup_y'].split(' ')
            subgroup_set_2 = [ int(s) for s in subgroup_set_2 ]

        res = subgroup_set_1 + subgroup_set_2
        idx = np.unique(res, return_index = True)[1]
        # print( idx )
        res = [ res[i] for i in sorted( idx ) ]
        if len( res ) < 50:
            for s, c in OrderedDict(sorted((_.items()), key=lambda x: x[1], reverse=True)).items():
                # print( s, c )
                # input('>')
                if s not in res:
                    res += [s]
                if len(res) == 50:
                    break
        res = [ str(r) for r in res ]
        sg.append( " ".join( res ) )

    df['subgroup_x'] = sg
    df = df.drop('subgroup_y', axis = 1)
    df.rename( columns = { 'subgroup_x': 'subgroup' }, inplace = True )
    print( df )

    df.to_csv('./subgroup.csv', index = 0)

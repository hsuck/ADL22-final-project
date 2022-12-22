import pandas as pd
from ml_metrics import mapk

def evaluate_map( actual_csv: str, pred_csv: str ):
    act = pd.read_csv(actual_csv).fillna('')
    pred = pd.read_csv(pred_csv).fillna('')

    assert len(act) == len(pred), f"act, pred length: {len(act)}, {len(pred)}"
    assert 'user_id' in act.columns
    assert 'course_id' in act.columns or 'subgroup' in act.columns
    assert len(act.columns) == 2
    assert (act.columns == pred.columns).all()

    col = 'course_id' if 'course_id' in act.columns else 'subgroup'

    act = act.sort_values(by='user_id')
    pred = pred.sort_values(by='user_id')

    for i in range( len(act) ):
        assert act['user_id'][i] == pred['user_id'][i]

    return mapk(
        actual = [ x.split(' ') for x in act[col] ],
        predicted = [ x.split(' ') for x in pred[col] ],
        k=50
    )


    

import pandas as pd
import numpy as np
from scipy import stats
from controls import *

def load_train_csv(sid, amt=None, save=False):
    train_df = pd.read_csv(f'data/numerai_dataset_{sid}/numerai_training_data.csv')
    ignored = ['id', 'era', 'data_type', 'target']

    columns = [c for c in train_df.columns if c not in ignored]

    X = np.array([train_df[c].tolist() for c in columns]).T[:amt]
    eras = train_df['era'].tolist()[:amt]
    y = np.array(train_df['target'].tolist())[:amt]
    print(set(train_df['era'].tolist()))
    amt = 'all' if amt == None else amt
    if save:
        dump_bin(f'temp_sets/train_{amt}.pkl', (X, y, eras))
    return X, y, eras


def load_live_csv(sid, save=False, chunksize=10**5, filter=None):
    ignored = ['id', 'era', 'data_type', 'target']
    X, y = [], []
    ids = []
    eras = []
    for chunk in pd.read_csv(f'data/numerai_dataset_{sid}/numerai_tournament_data.csv', chunksize=chunksize):
        if filter == None:

            live_df = chunk
            live_df = live_df.drop(columns=['target',])
        else:
            live_df = chunk.where(chunk['data_type'] == filter)

        live_df = live_df.dropna()
        columns = [c for c in live_df.columns if c not in ignored]
        if live_df.shape[0] != 0:
            ids.extend(live_df['id'].tolist())
            eras.extend(live_df['era'].tolist())
            x = np.array([live_df[c].tolist() for c in columns]).T
            if filter != None:
                y.extend(live_df['target'].tolist())
            if len(X) == 0:
                X = x
            else:
                X = np.concatenate((X, x))
    if save:
        if filter != None:
            dump_bin('temp_sets/val.pkl', (X, y, ids, eras,))
        else:
            dump_bin('temp_sets/live.pkl', (X, y, ids, eras,))
    return X, y, ids, eras

if __name__ == '__main__':

    load_live_csv(259, save=True, filter=None)
    #load_train_csv(259, save=True, amt=None)

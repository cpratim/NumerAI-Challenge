from numerapi import NumerAPI
from config import KEY, SECRET
from zipfile import ZipFile
import os
from data import load_train_csv, load_live_csv
from mode import ModeClustering
import pandas as pd
import os
from controls import *

client = NumerAPI(secret_key = SECRET, public_id = KEY)

def current_round():
    rounds = client.get_competitions()
    return rounds[0]['number']

def get_latest_dataset(path='data', fname=''):
    client.download_current_dataset(dest_path=path)
    for f in os.listdir(path):
        try:
            if f.split('.')[1] == 'zip':
                os.remove(f'{path}/{f}')
        except:
            pass

def upload_predictions(model, mid=None, round=current_round()):
    fname = f'predictions/round_{round}_model.csv'

    lX, y, ids, eras = load_live_csv(round)

    live_preds = model.predict(lX)
    pred_csv = pd.DataFrame({'id': ids, 'prediction': live_preds})
    pred_csv = pred_csv.set_index('id')
    pred_csv.to_csv(fname)
    client.upload_predictions(fname, model_id=mid)

def train_model(model, save=True, round=current_round()):

    tX, ty, teras = load_train_csv(round)
    model.fit(tX, ty, teras)
    if save:
        dump_bin(f'models/model_{round}.pkl', model)
    return model

if __name__ == '__main__':
    get_latest_dataset()
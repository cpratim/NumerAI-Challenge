import numpy as np
from time import time
from controls import *
from threading import Thread
from sklearn.model_selection import train_test_split
from mode import ModeClustering
from multiprocessing import Pool

corrs = []

def callback(p):

    model, xy, n = p
    print(f'Thread {n} started')
    X_train, X_test, y_train, y_test = xy
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    corr = np.corrcoef(y_pred, y_test,)[0, 1]
    corrs.append(corr)
    print(f'Thread [{n+1}] finished | Corr: {corr}')
    return corr

def avg_corr(model, X, y, samples=5, test_size=.2, disp=True):

    corrs = []
    Xy_map = [
        (model, train_test_split(X, y, test_size=test_size), i) for i in range(samples)
    ]
    pool = Pool(samples)
    result = pool.imap(callback, Xy_map)
    pool.close()
    pool.join()
    for c in result:
        corrs.append(c)
    return np.average(corrs)

if __name__ == '__main__':

    X, y = read_bin('temp_sets/train_50000.pkl')

    model = ModeClustering(
        n_feat_samples=150,
        n_target_acc=5,
    )
    avg_c = avg_corr(model, X, y, samples=5)

    print(f'Average Corr: {avg_c}')

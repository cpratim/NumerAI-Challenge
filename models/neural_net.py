from controls import *
from config import current_round
import numpy as np
from datetime import datetime
import os
import json
from random import randint
from warnings import filterwarnings
from pprint import pprint
from tqdm import tqdm
from math import floor

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
)

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def correlation(x, y):
    return np.corrcoef(x, y)[0, 1]


def log_model(model, corr, learning_rate, epochs, loss, optimizer, notes = None, log_dir="model_logs"):

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        with open(f"{log_dir}/logs.json", "w") as f:
            json.dump({}, f, indent=4)
        architecture = {}

    else:
        with open(f"{log_dir}/logs.json", "r") as f:
            architecture = json.load(f)

    layers = [l.strip() for l in str(model).split("  ")]
    name = layers[0][:-2]
    dtime = str(datetime.now())[:19]
    fname = "models/" + "".join([str(randint(0, 10)) for i in range(10)]) + ".model"
    torch.save(model, fname)
    architecture[dtime] = {
        "test_correlation": corr[0],
        "train_correlation": corr[1],
        "loss": loss,
        "optimizer": optimizer,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "file": fname,
        "architecture": {},
        "notes": notes,
    }
    for l in layers[1:]:
        l = l[:-2] if l == layers[-1] else l
        si = l.index(":")
        l_name = l[1 : si - 1]
        l_params = l[si + 2 :]
        architecture[dtime]["architecture"][l_name] = l_params

    with open(f"{log_dir}/logs.json", "w") as f:
        json.dump(architecture, f, indent=4)


class LinearNet(nn.Module):
    def __init__(self, n_feat, n_out=1):

        super().__init__()

        self.linear1 = nn.Linear(n_feat, 700)
        self.relu1 = nn.ReLU()
        self.rnn1 = nn.RNNCell(700, 200)
        self.linear2 = nn.Linear(200, n_out)

    def forward(self, x):

        x = self.linear1(x)
        x = self.relu1(x)
        x = self.rnn1(x)
        x = self.linear2(x)
        return x


def train(
    model,
    X, y,
    X_val, y_val,
    score_func=correlation,
    stop_streak=100,
    history=True,
    epochs=10000,
    batch_size=50000,
    learning_rate=1e-5,
    optimizer=optim.Adam,
    loss_func=nn.CrossEntropyLoss(),
):

    X = X.to(device)
    y = y.to(device)
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    splits = floor(len(X) / batch_size) if batch_size != None else 1

    X_batches = torch.tensor_split(X, splits)
    y_batches = torch.tensor_split(y, splits)
    
    max_score = 0
    score_history = []

    bar = tqdm(range(epochs))
    for epoch in bar:
        for x, y in zip(X_batches, y_batches):
            y_pred = model(x)
            loss = loss_func(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        #e_score = score_func(model(X_val).detach().cpu().numpy().reshape(-1,), y_val)
        bar.set_description(f"Score: {round(loss.item(), 5)}")
        #score_history.append(e_score)

    if history:
        return model, score_history
    return model



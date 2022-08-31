import torch
import torch.nn as nn
import json

DICT_LOSS = {
    "MSELoss": nn.MSELoss
}

DICT_OPTIMIZER = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD
}

LIST_LEN = [[1,2,3], [1,2,3], [1,2], [1,2,3], [1,2,3,4], [1,2]]

def split_and_one_hot(l):
    y = int(l[0])
    x = []
    for i in range(len(l[1:])):
        for j in LIST_LEN[i]:
            if j == int(l[1:][i]):
                x.append(1)
            else:
                x.append(0)
    return x, y

def compute_accuracy(pred_y, y):
    round_y = torch.round(pred_y)

    acc = 0
    for i in range(len(round_y)):
        if round_y[i] == y[i]:
            acc += 1
    acc = acc/len(round_y)
    
    return acc


def load_config(config_file="model_config.json"):
    with open(config_file) as f:
        conf = json.load(f)
    return conf


def configurazione_migliore(res_gridsearch):
    best_config = ""
    best_validation_acc = 0.0
    for k in res_gridsearch:
        if res_gridsearch[k]['validation_accuracy'] > best_validation_acc:
            best_config = k
            best_validation_acc = res_gridsearch[k]['validation_accuracy']

    return res_gridsearch[best_config]['comb']

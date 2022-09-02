import torch
import torch.nn as nn
import json

DICT_LOSS = {
    "MSELoss": nn.MSELoss,
    "CrossEntropyLoss": nn.CrossEntropyLoss
}

DICT_OPTIMIZER = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD
}


def compute_accuracy(pred_y, y):
    round_y = torch.round(pred_y)
    acc = 0
    for i in range(len(round_y)):
        acc += torch.dot(round_y[i], y[i])
    acc = acc/len(round_y)
    
    return acc.item()


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


import torch
import json
import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils import load_config, configurazione_migliore,  DICT_LOSS, DICT_OPTIMIZER, compute_accuracy
from dataset_wrapper import DatasetWrapper
from model import Model
from grid_search import gridSearch


config = load_config()


random.seed(config['seed'])
torch.manual_seed(config['seed'])

#data loading
training_dataset = DatasetWrapper("dataset/monks-"+config['monk']+".train")
test_dataset = DatasetWrapper("dataset/monks-"+config['monk']+".test")



#grid search phase
if(config['enable_grid_search']):
    res_gridsearch = gridSearch(Model, training_dataset, config)
    print(json.dumps(res_gridsearch, indent=4))

    config['configurazione_rete_finale'] = configurazione_migliore(res_gridsearch)

#training phase
best_model_parameters = config['configurazione_rete_finale']

best_model = Model(config['input_dim'], best_model_parameters[0], best_model_parameters[1], config["output_dim"])
loss_function = DICT_LOSS[config["loss_function"]]()
optimizer = DICT_OPTIMIZER[config["optimizer"]](best_model.parameters(), lr=best_model_parameters[3], momentum= best_model_parameters[4])

#training loop
training_acc = []
test_acc = []

if config['enable_per_epoch_test']:
    training_error = []
    test_error = []

dl_training =  DataLoader(training_dataset, batch_size=config['batch_size'])
dl_test =  DataLoader(test_dataset, batch_size=config['batch_size'])

for epoch in range(best_model_parameters[2]):

    epoch_acc_training = 0.0
    epoch_error_training = 0.0

    for x,y in dl_training:
        y_pred = best_model(x)
        loss_value = loss_function(y_pred, y)

        epoch_error_training += loss_value.item()
        epoch_acc_training += compute_accuracy(y_pred, y)


        #clean optimizer gradient values
        optimizer.zero_grad()
        #compute new gradient
        loss_value.backward()
        #update weights
        optimizer.step()
    

    training_acc.append(epoch_acc_training / len(dl_training))
    training_error.append(epoch_error_training / len(dl_training))


    #fase di test per epoca
    if config['enable_per_epoch_test']:
        epoch_acc_test = 0.0
        epoch_error_test = 0.0

        for x,y in dl_test:
            y_pred = best_model(x)
            loss_value = loss_function(y_pred, y)

            epoch_error_test += loss_value.item()
            epoch_acc_test += compute_accuracy(y_pred, y)


            #clean optimizer gradient values
            optimizer.zero_grad()
            #compute new gradient
            loss_value.backward()
            #update weights
            optimizer.step()
        
        test_acc.append(epoch_acc_test / len(dl_test))
        test_error.append(epoch_error_test / len(dl_test))




if config['enable_per_epoch_test']:
    
    
    print("Training Accuracy after {} epochs: {}".format(best_model_parameters[2], training_acc[-1]))
    print("Test Accuracy after {} epochs: {}".format(best_model_parameters[2], test_acc[-1]))
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=2)

    ax1[0].plot(training_acc)
    ax1[1].plot(training_error)
    
    ax2[0].plot(test_acc)
    ax2[1].plot(test_error)
    plt.show()





import itertools

from torch.utils.data import DataLoader
from sklearn.model_selection import KFold


from utils import compute_accuracy, DICT_LOSS, DICT_OPTIMIZER




def gridSearch(Model, dataset, config):

    input_dim = config["input_dim"]
    output_dim = config["output_dim"]


    
    first_hidden_layer_values = config['grid_search']['first_hidden_layer']
    second_hidden_layer_value = config['grid_search']['second_hidden_layer']
    epochs = config['grid_search']['epochs']
    learning_rate = config['grid_search']['learning_rate']
    momentum = config['grid_search']['momentum']

    combinations = itertools.product(*[first_hidden_layer_values, second_hidden_layer_value, epochs, learning_rate, momentum])

    res = {}
    if config["enable_kfold"]:
        kfold = KFold(n_splits=config["no_fold"], shuffle=True, random_state = config['seed'])
        
        for i, comb in enumerate(combinations):
            training_acc = 0.0
            validation_acc = 0.0

            for train_ids, validation_ids in kfold.split(dataset):
                dl_training = DataLoader(dataset.subset_from_indexes(train_ids), batch_size=config['batch_size'])
                dl_validation = DataLoader(dataset.subset_from_indexes(validation_ids), batch_size=config['batch_size'])

                res[str(i)] = {}
                m = Model(input_dim, comb[0], comb[1], output_dim)
                loss_function = DICT_LOSS[config["loss_function"]]()
                optimizer = DICT_OPTIMIZER[config["optimizer"]](m.parameters(), lr=comb[3], momentum=comb[4])
                
                #training loop
                for e in range(comb[2]):
                    
                    for x,y in dl_training:
                        y_pred = m(x)
                        loss_value = loss_function(y_pred, y)

                        #clean optimizer gradient values
                        optimizer.zero_grad()
                        #compute new gradient
                        loss_value.backward()
                        #update weights
                        optimizer.step()

                #testing 
                training_accuracy = 0.0
                for x,y in dl_training:
                    y_pred = m(x)
                    acc = compute_accuracy(y_pred, y)
                    training_accuracy += acc
                training_acc += training_accuracy/len(dl_training)

                validation_accuracy = 0.0
                for x,y in dl_validation:
                    y_pred = m(x)
                    acc = compute_accuracy(y_pred, y)
                    validation_accuracy += acc
                validation_acc += validation_accuracy/len(dl_validation)

            res[str(i)]['comb'] = comb
            res[str(i)]['training_accuracy'] = training_acc/config["no_fold"]
            res[str(i)]['validation_accuracy'] = validation_acc/config["no_fold"]
            

    else:
        validation_size = int(len(dataset)*config['validation_fraction'])
        training_dataset, validation_dataset  = dataset.train_validation_split(validation_size, shuffle=True)
        
        for i, comb in enumerate(combinations):
            res[str(i)] = {}
            res[str(i)]['comb'] = comb
            m = Model(input_dim, comb[0], comb[1], output_dim)
            loss_function = DICT_LOSS[config["loss_function"]]()
            optimizer = DICT_OPTIMIZER[config["optimizer"]](m.parameters(), lr=comb[3], momentum=comb[4])

            dl_training = DataLoader(training_dataset, batch_size=config['batch_size'])
            #training loop
            for e in range(comb[2]):
                
                for x,y in dl_training:
                    y_pred = m(x)
                    loss_value = loss_function(y_pred, y)

                    #clean optimizer gradient values
                    optimizer.zero_grad()
                    #compute new gradient
                    loss_value.backward()
                    #update weights
                    optimizer.step()
            
            training_accuracy = 0
            for x,y in dl_training:
                y_pred = m(x)
                acc = compute_accuracy(y_pred, y)
                training_accuracy += acc
            training_accuracy = training_accuracy/len(dl_training)
            res[str(i)]['training_accuracy'] = training_accuracy


            dl_validation = DataLoader(validation_dataset, batch_size=config['batch_size'])
            validation_accuracy = 0
            for x,y in dl_validation:
                y_pred = m(x)
                acc = compute_accuracy(y_pred, y)
                validation_accuracy += acc
            validation_accuracy = validation_accuracy/len(dl_validation)
            res[str(i)]['validation_accuracy'] = validation_accuracy
    
    return res




        

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

#  these work on windows, but not on ubuntu
#  from src.data_loading.process_inputs import parse_config
#  from src.data_loading.Dataset import Dataset
#  from src.models.neural_net import PcNet, EmbeddingReducingNN
#  from src.models.training import Trainer, Tester, ModelManager
#  from src.performance_evaluation.standard_error_computation import calculate_standard_error_by_bootstrapping

from data_loading.process_inputs import parse_config
from data_loading.Dataset import Dataset
from models.neural_net import PcNet, PcNet_chemBERTa, PcNet_RDKit, EmbeddingReducingNN
from models.training import Trainer, Tester, ModelManager
from performance_evaluation.standard_error_computation import calculate_standard_error_by_bootstrapping

import random
from tqdm import tqdm

import os
import time  # for timestamps to easily distinguish results

#######################################################################################################################
# #######################################ini file  Parser for dataset selection########################################

data_used, use_model, files, do_regression, nr_prediction_classes = parse_config()
print(parse_config())

data_set = Dataset(files['embeddings'],
                   files['compound_vectors'],
                   files['label_file'],
                   data_used[0],
                   files['cluster_map'] if files['cluster_map'] != "" else None)

class_borders = np.linspace(data_set.data_ranges[data_set.data_type][0], data_set.data_ranges[data_set.data_type][1],
                            nr_prediction_classes+1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#######################################################################################################################
# ################################################parameters and hyper parameters######################################

number_of_random_draws = 20
#  number_of_random_draws = 2
batch_sizes = list(range(10, 1024, 5))
#  batch_sizes = list(range(10, 2048, 5))
learning_rates = [0.01, 0.001, 0.0001]

#  numbers_of_epochs = list(range(1, 3))
numbers_of_epochs = list(range(100, 300))

#######################################################################################################################
# ###############################################################train/test split######################################

train_data_split = []
nr_training_splits = 5
#  nr_training_splits = 2

train_rest, test_split = train_test_split(data_set, test_size=1 / (nr_training_splits+1), random_state=42)
all_training_samples = train_rest
for i in range(nr_training_splits, 1, -1):
    train_rest, train_split = train_test_split(train_rest, test_size=1 / i, random_state=42)
    train_data_split.append(train_split)
train_data_split.append(train_rest)

#######################################################################################################################
# ###############################################################tuning################################################

t = time.localtime()
timestamp = time.strftime('%b-%d-%Y_%H%M', t)
os.mkdir("../Results/Results_"+timestamp)


best_parameters_overall = [0, 0, 0]

current_best_r2m = 0

for test_train_index in tqdm(range(nr_training_splits)):
    print('Using device:', device)
    # print(torch.cuda.memory_summary(device,abbreviated=False))############################################
    for optimization in tqdm(range(number_of_random_draws)):
        if use_model == "chemVAE":
            model = PcNet()
        elif use_model == "chemBERTa":
            model = PcNet_chemBERTa()
        elif use_model == "RDKit":
            model = PcNet_RDKit()
        # model = EmbeddingReducingNN()
        batch_size = random.choice(batch_sizes)
        learning_rate = random.choice(learning_rates)
        number_of_epochs = random.choice(numbers_of_epochs)

        tester = Tester(device, timestamp)
        trainer = Trainer(device, optim.Adam(model.parameters(), lr=learning_rate), nr_prediction_classes,
                          class_borders, data_used[0], timestamp)

        # create n train and test sets
        training_sets = []
        testing_tests = []

        for i in range(len(train_data_split)):
            train_dataset = []
            test_dataset = train_data_split[i]
            for j in range(len(train_data_split)):
                # if i != j, the split is added to the training data, the case i == j is used as corresponding test set
                if j != i:
                    train_dataset += train_data_split[j]
            training_sets.append(torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                             shuffle=False))
            testing_tests.append(torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                                             shuffle=False))

        trainer.train(model, training_sets[test_train_index], number_of_epochs, batch_size)
        performance_regression = tester.test(model, testing_tests[test_train_index], data_used)
        if performance_regression > current_best_r2m:
            current_best_r2m = performance_regression
            best_parameters_overall = [batch_size, learning_rate, number_of_epochs]

print('Finished Tuning')
print(current_best_r2m)
print(best_parameters_overall)

#######################################################################################################################
# ###############################################################training##############################################

if use_model == "chemVAE":
    model = PcNet()
elif use_model == "chemBERTa":
    model = PcNet_chemBERTa()
elif use_model == "RDKit":
    model = PcNet_RDKit()
# model = EmbeddingReducingNN()
trainer = Trainer(device, optim.Adam(model.parameters(), lr=best_parameters_overall[1]), nr_prediction_classes,
                  class_borders, data_used[0], timestamp)
tester = Tester(device, timestamp)
model_manager = ModelManager(model, trainer, tester)

train_loader = torch.utils.data.DataLoader(dataset=all_training_samples, batch_size=best_parameters_overall[0],
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_split, batch_size=best_parameters_overall[0], shuffle=False)

model_manager.train(train_loader, best_parameters_overall[2], best_parameters_overall[0], tuning=False)
print('Finished Training')


model_manager.save_model(os.path.join("../Results/Results_"+timestamp+"/model_"+data_used[0]+"_"+timestamp+'.pth'))

model.load_state_dict(torch.load(os.path.join("../Results/Results_"+timestamp+"/model_" +
                                              data_used[0]+"_"+timestamp+'.pth')))


#######################################################################################################################
# ###############################################################testing###############################################

calculate_standard_error_by_bootstrapping(model_manager, test_loader, test_split, best_parameters_overall[0], data_used,
                                          timestamp)

print("Best r2m was: ", current_best_r2m)
print("Best parameters were:", best_parameters_overall)

"""
model = EmbeddingReducingNN()
model.load_state_dict(torch.load(os.path.join('..', 'model.pth')))
model_manager = ModelManager(model, None, None)
model_manager.predict(test_split, device)
"""

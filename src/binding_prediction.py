import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split

from data_loading.process_inputs import parse_config
from data_loading.Dataset import Dataset
from models.neural_net import PcNet, PcNet_chemBERTa, PcNet_RDKit  # , EmbeddingReducingNN
from models.training import Trainer, Tester, ModelManager
from performance_evaluation.standard_error_computation import run_test_and_calculate_standard_error_by_bootstrapping
from performance_evaluation.stats_and_output import *
from ast import literal_eval

import random
from tqdm import tqdm

import os
import time  # for timestamps to easily distinguish results

#######################################################################################################################
# ini file  Parser for dataset selection

data_used, use_model, files, do_regression, nr_prediction_classes, shuffle_drugs, shuffle_targets, dummy_run, \
overtrain, special_params = parse_config()

data_set = Dataset(files['embeddings'],
                   files['compound_vectors'],
                   files['label_file'],
                   data_used[0],
                   files['cluster_map'] if files['cluster_map'] != "" else None,
                   shuffle_drugs,
                   shuffle_targets)

class_borders = np.linspace(data_set.data_ranges[data_set.data_type][0], data_set.data_ranges[data_set.data_type][1],
                            nr_prediction_classes + 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#######################################################################################################################
# parameters and hyper parameters

# parameters for improving the code
# dummy_run: if True a dummy run to fix bugs is started
# and requires dummy_run = False, and existing hyper parameters from a previous run


if not dummy_run and not overtrain:
    number_of_random_draws = 20  # usually 10
elif not dummy_run and overtrain:
    number_of_random_draws = 1  # since we still want the best validation loss for the plots
elif dummy_run and overtrain:
    raise Exception("Overtraining only works with dummy_run set to False.")
else:
    number_of_random_draws = 2

# batch_sizes = list(range(128, 513, 4))  # for Davis
batch_sizes = list(range(128, 1025, 4))   # for BDB
learning_rates = [0.01, 0.001, 0.0001]

#  learning_rates = list(np.arange(0.0001, 0.01, 0.0001))

if not dummy_run:
    numbers_of_epochs = list(range(200, 401))
else:
    numbers_of_epochs = list(range(3, 6))

#######################################################################################################################
# train/test split

train_data_split = []

if not dummy_run:
    number_of_splits = 5  # three for training, one for validation, one for testing
else:
    number_of_splits = 2

train_rest, test_split = train_test_split(data_set, test_size=1 / (number_of_splits + 1), random_state=42)
all_training_samples = train_rest
for i in range(number_of_splits, 1, -1):
    train_rest, train_split = train_test_split(train_rest, test_size=1 / i, random_state=42)
    train_data_split.append(train_split)
train_data_split.append(train_rest)

#######################################################################################################################
# tuning

t = time.localtime()
timestamp = time.strftime('%b-%d-%Y_%H%M', t)
os.mkdir("../Results/Results_" + timestamp)

print('Using device:', device)

if not overtrain:
    best_parameters_overall = [0, 0, 0]
    best_validation_loss = np.inf

    for test_train_index in tqdm(range(number_of_splits)):
        for optimization in tqdm(range(number_of_random_draws)):

            if use_model == "chemVAE":
                model = PcNet()
            elif use_model == "chemBERTa":
                model = PcNet_chemBERTa()
            elif use_model == "RDKit":
                model = PcNet_RDKit()
            else:
                raise Exception("Model is undefined.")
            # model = EmbeddingReducingNN()

            batch_size = random.choice(batch_sizes)
            learning_rate = random.choice(learning_rates)
            number_of_epochs = random.choice(numbers_of_epochs)

            tester = Tester(device, timestamp)
            trainer = Trainer(device, optim.Adam(model.parameters(), lr=learning_rate), nr_prediction_classes,
                              class_borders, data_used[0], timestamp)

            # create n train and validation sets
            training_sets = []
            validation_set = []

            for i in range(len(train_data_split)):
                train_dataset = []
                test_dataset = train_data_split[i]
                for j in range(len(train_data_split)):
                    # if i != j
                    # the split is added to the training data, the case i == j is used as corresponding test set
                    if j != i:
                        train_dataset += train_data_split[j]
                training_sets.append(torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                                 shuffle=False))
                validation_set.append(torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                                                  shuffle=False))

            training_loss_per_epoch, validation_loss_per_epoch = trainer.train(model, training_sets[test_train_index],
                                                                               validation_set[test_train_index],
                                                                               number_of_epochs,
                                                                               batch_size, final_training=False)

            if validation_loss_per_epoch[-1] < best_validation_loss:
                best_validation_loss = validation_loss_per_epoch[-1]
                best_parameters_overall = [batch_size, learning_rate, number_of_epochs]

    print('Finished Tuning')
    print(best_parameters_overall)

else:
    print("Overtraining, skipped Tuning")
    best_parameters_overall = literal_eval(special_params['overtrain_params'])

#######################################################################################################################
# training

if use_model == "chemVAE":
    model = PcNet()
elif use_model == "chemBERTa":
    model = PcNet_chemBERTa()
elif use_model == "RDKit":
    model = PcNet_RDKit()
else:
    raise Exception("Model is undefined.")
# model = EmbeddingReducingNN()
trainer = Trainer(device, optim.Adam(model.parameters(), lr=best_parameters_overall[1]), nr_prediction_classes,
                  class_borders, data_used[0], timestamp)
tester = Tester(device, timestamp)
model_manager = ModelManager(model, trainer, tester)

train_loader = torch.utils.data.DataLoader(dataset=all_training_samples, batch_size=best_parameters_overall[0],
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_split, batch_size=best_parameters_overall[0], shuffle=False)

zeroes_predicted = True  # TODO: find a more elegant solution to prevent empty models, also this might not work.
can_this_cause_an_endless_loop = 0
training_loss_per_epoch = []
validation_loss_per_epoch = []
while zeroes_predicted:
    can_this_cause_an_endless_loop += 1
    if can_this_cause_an_endless_loop > 5:
        raise Exception("The model appears to predict zeroes after each new training.")
    print("Training attempt No: " + str(can_this_cause_an_endless_loop))
    training_loss_per_epoch, validation_loss_per_epoch = model_manager.train(train_loader, test_loader,
                                                                             best_parameters_overall[2],
                                                                             best_parameters_overall[0],
                                                                             final_training=True)

    if validation_loss_per_epoch[-1] != 0:  # Sometimes there is a model that predicts only zeroes on the testing data.
        zeroes_predicted = False

print('Finished Training')
if not training_loss_per_epoch:
    raise Exception("Something went wrong. The training loss per epoch is empty.")

print_loss_per_epoch(validation_loss_per_epoch, training_loss_per_epoch, data_used[0], timestamp)

model_manager.save_model(
    os.path.join("../Results/Results_" + timestamp + "/model_" + data_used[0] + "_" + timestamp + '.pth'))

model.load_state_dict(torch.load(os.path.join("../Results/Results_" + timestamp + "/model_" +
                                              data_used[0] + "_" + timestamp + '.pth')))

#######################################################################################################################
# testing

# in a sub-function the function that creates the scatter plot is called
run_test_and_calculate_standard_error_by_bootstrapping(model_manager, test_loader, test_split,
                                                       best_parameters_overall, data_used, timestamp)

print("Best parameters were:", best_parameters_overall)

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split

from data_loading.process_inputs import parse_config
from data_loading.Dataset import Dataset
from models.neural_net import PcNet, PcNet_chemBERTa, PcNet_RDKit  # , EmbeddingReducingNN
from models.training import Trainer, Tester, ModelManager
from performance_evaluation.standard_error_computation import run_test_and_calculate_standard_error_by_bootstrapping
from performance_evaluation.stats_and_output import *

import random
from tqdm import tqdm

import os
import time  # for timestamps to easily distinguish results

#######################################################################################################################
# ini file  Parser for dataset selection

data_used, use_model, files, do_regression, nr_prediction_classes, shuffle_drugs, shuffle_targets = parse_config()
print(parse_config())

data_set = Dataset(files['embeddings'],
                   files['compound_vectors'],
                   files['label_file'],
                   data_used[0],
                   files['cluster_map'] if files['cluster_map'] != "" else None,
                   shuffle_drugs,
                   shuffle_targets)

class_borders = np.linspace(data_set.data_ranges[data_set.data_type][0], data_set.data_ranges[data_set.data_type][1],
                            nr_prediction_classes+1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#######################################################################################################################
# parameters and hyper parameters

# parameters for improving the code
true_run = True  # if False a dummy run to observe bugs is started
overtrain = False  # adds 100 epochs to validation and training

if true_run:
    number_of_random_draws = 1  # TODO: if not 10 turn to 10
else:
    number_of_random_draws = 2

batch_sizes = list(range(10, 256, 5))
#  batch_sizes = list(range(10, 2048, 5))
learning_rates = [0.01, 0.001, 0.0001]
#  learning_rates = list(np.arange(0.0001, 0.01, 0.0001))

if true_run:
    numbers_of_epochs = list(range(100, 301))
else:
    numbers_of_epochs = list(range(3, 6))


#######################################################################################################################
# train/test split

train_data_split = []

if true_run:
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
os.mkdir("../Results/Results_"+timestamp)


best_parameters_overall = [0, 0, 0]

current_best_r2m = 0

losses_per_epoch = []
best_loss_per_epoch = []

print('Using device:', device)

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

        #batch_size = random.choice(batch_sizes)
        #learning_rate = random.choice(learning_rates)
        #number_of_epochs = random.choice(numbers_of_epochs)

        # TODO: remove experiment
        batch_size = 235
        learning_rate = 0.01
        number_of_epochs = 196

        tester = Tester(device, timestamp)
        trainer = Trainer(device, optim.Adam(model.parameters(), lr=learning_rate), nr_prediction_classes,
                          class_borders, data_used[0], timestamp)

        # create n train and test sets
        training_sets = []
        validation_set_ = []  # isn't this a validation set?

        for i in range(len(train_data_split)):
            train_dataset = []
            test_dataset = train_data_split[i]
            for j in range(len(train_data_split)):
                # if i != j, the split is added to the training data, the case i == j is used as corresponding test set
                if j != i:
                    train_dataset += train_data_split[j]
            training_sets.append(torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                             shuffle=False))
            validation_set_.append(torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                                               shuffle=False))

        new_loss_per_epoch = trainer.train(model, training_sets[test_train_index], number_of_epochs, batch_size,
                                           final_training=False)
        # losses_per_epoch += [new_loss_per_epoch]  # TODO: maybe keeping all loses per epoch is useful

        # uses r2m as cutoff for best parameters
        performance_regression = tester.test(model, validation_set_[test_train_index], data_used, [0, 0, 0])
        # best parameters are not used here therefore the last parameter is just zeroes
        if performance_regression > current_best_r2m:
            current_best_r2m = performance_regression
            best_loss_per_epoch = new_loss_per_epoch
            best_parameters_overall = [batch_size, learning_rate, number_of_epochs]
        # TODO: maybe find an alternative for deciding which model was best
        # if len(best_loss_per_epoch) == 0 or (statistics.mean(new_loss_per_epoch[-50:]) < (statistics.mean(
        #        best_loss_per_epoch[-50:]))):
        #    best_loss_per_epoch = new_loss_per_epoch
        #    best_parameters_overall = [batch_size, learning_rate, number_of_epochs]
        # print(performance_regression)
        print(best_parameters_overall)

print('Finished Tuning')
print(current_best_r2m)
print(best_parameters_overall)

#######################################################################################################################
# training

# best_parameters_overall = [190, 0.0001, 295]

if overtrain:
    best_parameters_overall[2] += 100

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
while zeroes_predicted:
    can_this_cause_an_endless_loop += 1
    if can_this_cause_an_endless_loop > 5:
        raise Exception("The model appears to predict zeroes after each new training.")
    print("Training attempt No: "+str(can_this_cause_an_endless_loop))
    training_loss_per_epoch = model_manager.train(train_loader, best_parameters_overall[2], best_parameters_overall[0],
                                                  final_training=True)
    if model_manager.test(test_loader, data_used, best_parameters_overall, prevent_zeroes=True):
        zeroes_predicted = False

print('Finished Training')
if not training_loss_per_epoch:
    raise Exception("Something went wrong. The training loss per epoch is empty.")

model_manager.save_model(os.path.join("../Results/Results_"+timestamp+"/model_"+data_used[0]+"_"+timestamp+'.pth'))

model.load_state_dict(torch.load(os.path.join("../Results/Results_"+timestamp+"/model_" +
                                              data_used[0]+"_"+timestamp+'.pth')))

print_loss_per_epoch(best_loss_per_epoch, training_loss_per_epoch, data_used[0], timestamp)
# TODO: remove comment for true run

#######################################################################################################################
# testing

run_test_and_calculate_standard_error_by_bootstrapping(model_manager, test_loader, test_split,
                                                       best_parameters_overall, data_used, timestamp)

print("Best r2m was: ", current_best_r2m)
# TODO: remove comment for true run
print("Best parameters were:", best_parameters_overall)

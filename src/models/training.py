from performance_evaluation.stats_and_output import *
import torch


class Trainer:
    def __init__(self, device, optimizer, nr_classes, class_borders, data_used, timestamp):
        self.device = device
        self.optimizer = optimizer
        self.criterion0 = torch.nn.MSELoss(reduction='sum')
        self.nr_classes = nr_classes
        if self.nr_classes == 2:
            self.criterion1 = torch.nn.BCELoss()
        else:
            self.criterion1 = torch.nn.MSELoss(reduction='sum')
            self.class_borders = class_borders
        self.data_used = data_used  # this and the timestamp is only to order files if multiple models are run
        self.timestamp = timestamp

    def _map_regression_list_to_classes(self, regression_list):
        output_list = torch.empty(regression_list.shape)
        for j in range(len(regression_list)):
            label = regression_list[j]
            for i in range(1, len(self.class_borders)):
                lower = self.class_borders[i-1]
                upper = self.class_borders[i]
                if label == max(self.class_borders):
                    output_list[j] = (max(self.class_borders) + max(self.class_borders[:-1])) / 2
                elif lower <= label < upper:
                    output_list[j] = (upper+lower)/2

        return output_list

    def train(self, model, data_for_training, data_for_validating, number_of_epochs, batch_size,
              final_training):
        validation_loss_per_epoch = []
        training_loss_per_epoch = []
        for epoch_index in range(number_of_epochs):
            training_loss = 0.0
            validation_loss = 0.0

            for i, (protein_compounds, regression_label, _, _) in enumerate(data_for_training):
                self.optimizer.zero_grad()
                inputs = protein_compounds
                regression_label = regression_label.unsqueeze(1).to(self.device).double()
                regression_output, class_output = model(inputs)
                loss = self.criterion0(regression_output.double().to(self.device), regression_label)
                total_loss = loss
                total_loss.backward()
                self.optimizer.step()
                training_loss += total_loss.item()

            # training_loss_per_epoch += [training_loss/batch_size]
            training_loss_per_epoch += [training_loss / (batch_size*len(data_for_training))]

            for i, (protein_compounds, regression_label, _, _) in enumerate(data_for_validating):
                # self.optimizer.zero_grad()
                inputs = protein_compounds
                regression_label = regression_label.unsqueeze(1).to(self.device).double()
                regression_output, class_output = model(inputs)
                loss = self.criterion0(regression_output.double().to(self.device), regression_label)
                total_loss = loss
                # total_loss.backward()
                # self.optimizer.step()
                validation_loss += total_loss.item()

            # validation_loss_per_epoch += [validation_loss / batch_size]
            validation_loss_per_epoch += [validation_loss / (batch_size*len(data_for_validating))]

            if final_training:
                # print('[%d] loss: %.7f' % (epoch_index + 1, validation_loss / batch_size))
                print('[%d] loss: %.7f' % (epoch_index + 1, validation_loss / (batch_size*len(data_for_validating))))

        return training_loss_per_epoch, validation_loss_per_epoch


class Tester:
    def __init__(self, device, timestamp):
        self.device = device
        self.timestamp = timestamp

    def _collect_predictions_and_labels(self, model, protein_compound, labels, name_pairs, all_labels, all_predicted,
                                        all_name_pairs, i):

        outputs = model(protein_compound)[i].double().to(self.device)
        # print(protein_compound)
        outputs = outputs.tolist()
        outputs = [j[0] for j in outputs]
        # print(sum(outputs))  # TODO: find out why this sometimes becomes zeroes
        outputs = torch.tensor(outputs, dtype=torch.float64)
        all_labels += labels.tolist()
        all_predicted += outputs.tolist()
        for i in range(len(name_pairs[0])):
            all_name_pairs.append((name_pairs[0][i], name_pairs[1][i]))

    def test(self, model, data_for_testing, data_used, best_parameters_overall, final_prediction=False, bootstrap=False,
             nr_of_hard_samples=1):
        all_regression_labels = []
        all_regression_predicted = []
        all_name_pairs = []

        all_binary_labels = []
        all_binary_predicted = []
        with torch.no_grad():
            for data in data_for_testing:
                protein_compound, regression_labels, class_labels, pair_names = data

                self._collect_predictions_and_labels(model, protein_compound, regression_labels, pair_names,
                                                     all_regression_labels, all_regression_predicted, all_name_pairs, 0)
                #  self._collect_predictions_and_labels(model, protein_compound, class_labels, all_binary_labels,
                #                                     all_binary_predicted, 1)

        if bootstrap:
            return bootstrap_stats(all_regression_predicted, all_regression_labels, data_used)
                   #bootstrap_stats(all_binary_predicted, all_binary_labels, data_used)

        if final_prediction:
            plot_output(all_regression_predicted, all_regression_labels, data_used, self.timestamp,
                        plot_name='scatter_plot_regression_'+data_used[0]+"_"+self.timestamp+".png")
            #  plot_output(all_binary_predicted, all_binary_labels, data_used, plot_name='scatter_plot_classes.png')
            print_stats(all_regression_predicted, all_regression_labels, data_used, best_parameters_overall,
                        self.timestamp)
            #  print_stats(all_binary_predicted, all_binary_labels, data_used)

            plot_output_hist(all_regression_predicted, self.timestamp,
                             plot_name='hist_after_prediction_'+data_used[0]+"_"+self.timestamp+".png")

        else:
            # print(all_regression_labels)
            # print(all_regression_predicted)
            return emetrics.get_r2m(all_regression_labels, all_regression_predicted)
                   #emetrics.get_r2m(all_binary_labels, all_binary_predicted)

        if nr_of_hard_samples > 0:
            find_hardest_samples(all_regression_predicted, all_regression_labels, all_name_pairs, nr_of_hard_samples)


class ModelManager:
    def __init__(self, model, trainer: Trainer, tester: Tester):
        self.model = model
        self.trainer = trainer
        self.tester = tester

    def train(self, data_for_training, data_for_testing, amount_of_epochs, batch_size, final_training):
        loss_per_epoch = self.trainer.train(self.model, data_for_training, data_for_testing, amount_of_epochs,
                                            batch_size, final_training)
        return loss_per_epoch

    def test(self, data_for_testing, data_used, best_parameters_overall, final_prediction=False, bootstrap=False):
        return self.tester.test(self.model, data_for_testing, data_used, best_parameters_overall, final_prediction,
                                bootstrap)

    def save_model(self, file_path):
        self.model.save(file_path)

    def predict(self, data_to_predict, device):
        all_predicted = []
        all_name_pairs = []
        with torch.no_grad():
            for data in data_to_predict:
                protein_compound, _, _, pair_names = data
                output = self.model(protein_compound)[0].double().to(device)

                all_predicted.append(output)
                all_name_pairs.append(pair_names)

        for i in range(len(all_name_pairs)):
            print(str(all_name_pairs[i]) + '\t' + str(all_predicted[i].item()))

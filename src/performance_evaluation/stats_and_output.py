import matplotlib.pyplot as plt
# import src.performance_evaluation.emetrics as emetrics  # from Hakime
import performance_evaluation.emetrics as emetrics  # from Hakime
import lifelines.utils
import numpy as np
import os


def pltcolor(lst):
    colors = []
    for item in lst:
        if item <= 0.5:
            colors.append('cornflowerblue')
        else:
            colors.append('red')
    return colors


def print_loss_per_epoch(validation_loss_vector, training_loss_vector, data_used, timestamp):
    # print(validation_loss_vector)
    # print(training_loss_vector)
    validation_loss_vector = validation_loss_vector[1:]
    training_loss_vector = training_loss_vector[1:]
    plt.clf()
    x = np.arange(len(validation_loss_vector))
    ax = plt.subplot(111)
    ax.plot(x, validation_loss_vector, label='Validation Loss')
    ax.plot(x, training_loss_vector, label='Training Loss')
    ax.legend()
    plt.title("Loss over Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("../Results/Results_"+timestamp+"/loss_over_epoch_"+data_used+"_"+timestamp+".png")


def plot_output(predicted, labels, data_used, timestamp, cols='cornflowerblue', plot_name="scatterplot.png"):
    plt.clf()
    plt.scatter(predicted, labels, alpha=0.2, color=cols, edgecolors="black")

    if data_used[0] == "pkd":
        plt.plot([4, 11], [4, 11], ls="--", c=".3")
        plt.xlim(4, 11)
        plt.ylim(4, 11)

    if data_used[0] == "kiba":
        plt.plot([7, 18], [7, 18], ls="--", c=".1")
        plt.xlim(7, 18)
        plt.ylim(7, 18)

    plt.title(data_used[1])
    plt.xlabel("Predicted")
    plt.ylabel("Measured")
    plt.savefig(os.path.join("../Results/Results_"+timestamp+"/", plot_name))


def plot_output_hist(predicted, timestamp, plot_name):
    plt.clf()
    plt.hist(predicted)
    plt.title("Interaction Values after predicting.")
    plt.xlabel("Values")
    plt.ylabel("Frequencies")
    plt.savefig(os.path.join("../Results/Results_" + timestamp + "/", plot_name))

    with open(os.path.join("../Results/Results_" + timestamp + "/", "predicted_values.txt"), "w") as g:
        for s in predicted:
            g.write(str(s) + " ")
            g.write("\n")


# TODO: Remove after fixing dependencies
# def only_r2m(all_predicted, all_labels):
#    return emetrics.get_r2m(all_labels, all_predicted)


def bootstrap_stats(all_predicted, all_labels, data_used):
    r2m = emetrics.get_r2m(all_labels, all_predicted)
    aupr = emetrics.compute_aupr(all_labels, all_predicted, data_used[0])
    ci = lifelines.utils.concordance_index(all_labels, all_predicted)
    mse = np.square(np.subtract(all_labels, all_predicted)).mean()

    return [r2m, aupr, ci, mse]


def print_stats(all_predicted, all_labels, data_used, timestamp):
    r2m = emetrics.get_r2m(all_labels, all_predicted)
    aupr = emetrics.compute_aupr(all_labels, all_predicted, data_used[0])
    ci = lifelines.utils.concordance_index(all_labels, all_predicted)
    mse = np.square(np.subtract(all_labels, all_predicted)).mean()

    print("The r2m value for this run is: ", round(r2m, 3))

    print("The AUPR for this run is: ", round(aupr, 3))

    print("The Concordance Index (CI) for this run is: ", round(ci, 3))

    print("The Mean Squared Error (MSE) for this run is: ", round(mse, 3))

    file1 = open("../Results/Results_"+timestamp+"/Results_"+data_used[0]+"_"+timestamp+".txt", "a")
    file1.write("The r2m value for this run is: "+str(round(r2m, 3))+"\n")
    file1.write("The AUPR for this run is: "+str(round(aupr, 3))+"\n")
    file1.write("The Concordance Index (CI) for this run is: "+str(round(ci, 3))+"\n")
    file1.write("The Mean Squared Error (MSE) for this run is: "+str(round(mse, 3))+"\n")
    file1.close()


def find_hardest_samples(all_predicted, all_labels, key_pairs, nr_of_samples):
    mean_square_error_list = []
    for i in range(len(all_predicted)):
        mean_square_error = np.square(all_labels[i] - all_predicted[i])
        keys = key_pairs[i]
        mean_square_error_list.append((mean_square_error, keys))

    mean_square_error_list.sort(key=lambda x: x[0], reverse=True)
    mean_square_error_list = mean_square_error_list[:nr_of_samples]
    # for entry in mean_square_error_list:
    # print(str(entry[1]) + '\n')

    return mean_square_error_list

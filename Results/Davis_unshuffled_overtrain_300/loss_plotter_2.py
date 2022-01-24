import matplotlib.pyplot as plt
import numpy as np

file = open("losses.txt", 'r')
content = file.readlines()

validation_loss_vector = content[0].split(" ")[:-1]
training_loss_vector = content[1].split(" ")[:-1]

validation_loss_vector = [float(x) for x in validation_loss_vector]
training_loss_vector = [float(x) for x in training_loss_vector]

def print_loss_per_epoch(validation_loss_vector, training_loss_vector):
    validation_loss_vector = validation_loss_vector[12:]
    training_loss_vector = training_loss_vector[12:]
    plt.clf()
    x = np.arange(len(validation_loss_vector))
    ax = plt.subplot(111)
    ax.plot(x, validation_loss_vector, label='Validation Loss')
    ax.plot(x, training_loss_vector, label='Training Loss')
    ax.legend()
    plt.title("Loss over Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("Loss.png")

print_loss_per_epoch(validation_loss_vector, training_loss_vector)
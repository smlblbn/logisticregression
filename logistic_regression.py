from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt


# Functions
def normalize(data):
    min_values = np.amin(data, axis=0)
    max_values = np.amax(data, axis=0)
    return (data - min_values) / (max_values - min_values)


def sigmoid(X, w):
    return 1 / (1 + np.exp(-np.dot(w.T, X)))


def cost(y, y_est):
    return (-1 / y.shape[0]) * np.sum(y * np.log(y_est) + (1 - y) * np.log(1 - y_est))


def grad(X, y, y_est):
    return (1 / y.shape[0]) * np.sum((y_est - y) * X, axis=1)


def accuracy(y, y_est):
    y_hat = (y_est >= 0.5).astype(int)
    loss01 = np.sum((y_hat != y).astype(int))
    return (1 - loss01 / y.shape[0]) * 100


# Import dataset (shape = (768,9))
dataset = np.genfromtxt('pima-indians-diabetes.csv', delimiter=',')

# Set seed and shuffle dataset
# SHUFFLE_SEED = 499
# np.random.seed(SHUFFLE_SEED)
# np.random.shuffle(dataset)

# Hyperparameters
SEED = 499
num_epoch = 10000
num_split = 5
alphas = [0.0003, 0.001, 0.1]
splits = [0, 134, 268, 402, 535, 668]

# Initialize the loss and accuracy result array
acc_train = []
acc_validation = []
J_train = []
J_validation = []

# Normalize and augment dataset
all_data = normalize(dataset[:, 0:8])
all_data = np.concatenate((np.ones((all_data.shape[0], 1), dtype=float), all_data), axis=1)

# Split the dataset into train and test parts
all_train_data = all_data[:668].T
test_data = all_data[668:].T

all_train_label = dataset[:668, 8]
test_label = dataset[668:, 8]

for i, alpha in enumerate(alphas):

    # Initialize weights
    np.random.seed(SEED)
    w = np.random.rand(9)

    # Initilaze the loss and accuracy temporary arrays
    _J_train = np.ndarray(shape=(num_split, num_epoch), dtype=float)
    _J_validation = np.ndarray(shape=(num_split, num_epoch), dtype=float)

    _acc_train = np.ndarray(shape=(num_split, num_epoch), dtype=float)
    _acc_validation = np.ndarray(shape=(num_split, num_epoch), dtype=float)

    for j in range(num_split):

        # Split the all train data into validation and train part for cross validation
        validation_data = all_data[splits[j]: splits[j + 1]].T
        validation_label = dataset[splits[j]: splits[j + 1], 8]

        train_data = np.concatenate((all_data[0: splits[j]], all_data[splits[j + 1]: 668]), axis=0).T
        train_label = np.concatenate((dataset[0: splits[j], 8], dataset[splits[j + 1]: 668, 8]), axis=0)

        for epoch in range(num_epoch):
            h_train = sigmoid(train_data, w)
            dw = grad(train_data, train_label, h_train)
            w = w - alpha * dw

            h_validation = sigmoid(validation_data, w)

            # Assign the value of loss and accuracy arrays
            _J_train[j, epoch] = cost(train_label, h_train)
            _J_validation[j, epoch] = cost(validation_label, h_validation)

            _acc_train[j, epoch] = accuracy(train_label, h_train)
            _acc_validation[j, epoch] = accuracy(validation_label, h_validation)

    # Append the mean of loss and accuracy arrays to alpha arrays
    J_train.append(np.mean(_J_train, axis=0))
    J_validation.append(np.mean(_J_validation, axis=0))

    acc_train.append(np.mean(_acc_train, axis=0))
    acc_validation.append(np.mean(_acc_validation, axis=0))

for i in range(len(alphas)):
    # Plots the loss array
    plt.plot(np.arange(num_epoch), J_train[i], label='train')
    plt.plot(np.arange(num_epoch), J_validation[i], label='validation')
    plt.title('Loss Functions for alpha = ' + str(alphas[i]))
    plt.legend(loc='upper right')
    plt.show()

    # Plots the accuracy array
    plt.plot(np.arange(num_epoch), acc_train[i], label='train')
    plt.plot(np.arange(num_epoch), acc_validation[i], label='validation')
    plt.title('Accuracy for alpha = ' + str(alphas[i]))
    plt.legend(loc='lower right')
    plt.show()

# Assigned best learning rate
_alpha = alphas[np.argmin(J_validation, axis=0)[-1]]

for epoch in range(num_epoch):
    h_all_train = sigmoid(all_train_data, w)
    dw = grad(all_train_data, all_train_label, h_all_train)
    w = w - _alpha * dw

h_test = sigmoid(test_data, w)

test_cost = cost(test_label, h_test)
test_accuracy = accuracy(test_label, h_test)

print(test_cost)
print(test_accuracy)

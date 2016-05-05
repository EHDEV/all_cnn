from __future__ import print_function
from lasagne_project_utils import load_data, shared_dataset
import theano
import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import scipy
import timeit
import inspect
from lasagne_project_nn import categorical_accuracy, all_CNN_C

def run_experiment(learning_rate=0.1, n_epochs=20, nkerns=[96, 192, 10],
             batch_size=200, verbose=False, kernel_shape=(3, 3)):
    """
    Wrapper function for testing Multi-Stage ConvNet on SVHN dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    """

    datasets = load_data()
    batch_size = 200

    rng = np.random.RandomState(23455)

    X_train, y_train = datasets[0]
    X_val, y_val = datasets[1]
    X_test, y_test = datasets[2]
    # X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    n_train_batches = X_train.get_value(borrow=True).shape[0]
    n_valid_batches = X_val.get_value(borrow=True).shape[0]
    n_test_batches = X_test.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    index = T.lscalar()  # index to a [mini]batch

    x = T.tensor4('x')
    y = T.ivector('y')

    channel = 3
    imsize = 32

    num_epochs = 12
    lambda_decay = 1e-3

    X_train = X_train.reshape((9000, 3, imsize, imsize))
    X_test = X_test.reshape((10000, 3, imsize, imsize))
    X_val = X_val.reshape((1000, 3, imsize, imsize))

    network = all_CNN_C(x)
    train_prediction = lasagne.layers.get_output(network)
    train_loss = lasagne.objectives.categorical_crossentropy(train_prediction, y)
    train_loss = train_loss.mean()

    # Regularization
    l2_penalty = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
    train_loss += lambda_decay * l2_penalty

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        train_loss, params, learning_rate=0.01, momentum=0.9)

    val_prediction = lasagne.layers.get_output(network)
    val_loss = categorical_accuracy(val_prediction, y)
    val_loss = val_loss.mean()
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network)
    test_loss = categorical_accuracy(test_prediction, y)

    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), y),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([index],
                       train_loss,
                       updates=updates,
                       givens={
                           x: X_train[index * batch_size: (index + 1) * batch_size],
                           y: y_train[index * batch_size: (index + 1) * batch_size]
                   }
            )

    val_fn = theano.function(
        [index],
        val_loss,
        givens={
            x: X_val[index * batch_size: (index + 1) * batch_size],
            y: y_val[index * batch_size: (index + 1) * batch_size]
        }
    )

    test_fn = theano.function(
        [index],
        [test_loss, test_acc],
        givens={
            x: X_test[index * batch_size: (index + 1) * batch_size],
            y: y_test[index * batch_size: (index + 1) * batch_size]
        }
    )

    train_nn(train_fn, val_fn, test_fn,
            n_train_batches, n_valid_batches, n_test_batches, num_epochs,
            verbose = True)


if __name__ == "__main__":
    run_experiment(verbose=True)

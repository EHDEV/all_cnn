import numpy
import theano
import theano.tensor as T
from lasagne_project_nn import LogisticRegression, HiddenLayer, LeNetConvPoolLayer, LeNetConvLayer, train_nn
from lasagne_project_utils import load_data, shared_dataset
import theano
from __future__ import print_function
import sys
import os
import time
import numpy as np
import numpy
import theano
import theano.tensor as T
import lasagne
import scipy
import timeit

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.eval().shape[0] == targets.eval().shape[0]
    if shuffle:
        indices = np.arange(inputs.eval().shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.eval().shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def all_cnn(verbose=False):

    datasets = load_data()
    rng = np.random.RandomState(23455)

    X_train, y_train = datasets[0]
    X_val, y_val = datasets[1]
    X_test, y_test = datasets[2]
    #X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    n_train_batches = X_train.get_value(borrow=True).shape[0]
    n_valid_batches = X_val.get_value(borrow=True).shape[0]
    n_test_batches = X_test.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size


    x = T.tensor4('x')
    y = T.ivector('y')

    channel = 3
    imsize= 32

    test_size = len(X_test.eval())
    train_size = len(X_train.eval())
    val_size = len(X_val.eval())

    num_epochs = 2

    X_train = X_train.reshape((test_size,3,32,32))
    X_test = X_test.reshape((train_size, 3, 32, 32))
    X_val = X_val.reshape((val_size, 3, 32, 32))

    network = build_cnn(x)

    train_prediction = lasagne.layers.get_output(network)
    train_loss = lasagne.objectives.categorical_crossentropy(train_prediction, y)
    train_loss = train_loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            train_loss, params, learning_rate=0.01, momentum=0.9)

    val_prediction = lasagne.layers.get_output(network)
    val_loss = lasagne.objectives.categorical_crossentropy(val_prediction, y)
    val_loss = val_loss.mean()
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, y)

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
            verbose=verbose)



if __name__ == "__main__":
    allCNN_C(verbose=True)

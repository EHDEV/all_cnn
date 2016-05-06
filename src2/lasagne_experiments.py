from __future__ import print_function
from lasagne_project_utils import load_data, shared_dataset
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne_project_nn import categorical_accuracy
from lasagne_project_nn import all_CNN_C, ConvPool_CNN_C, Strided_CNN_C, train_nn


def run_experiment(lr=0.01, num_epochs=50, nkerns=[96, 192, 10], lambda_decay=1e-3, conv_arch=all_CNN_C,
                   batch_size=200, verbose=False, kernel_shape=(3, 3)):
    """
    Wrapper function for testing Multi-Stage ConvNet on SVHN dataset

    :type lr: float
    :param lr: learning rate used (factor for the stochastic
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

    data_size = X_train.eval().shape[0]
    tdata_size = X_test.eval().shape[0]
    vdata_size = X_val.eval().shape[0]

    X_train = X_train.reshape((data_size, 3, imsize, imsize))
    X_test = X_test.reshape((tdata_size, 3, imsize, imsize))
    X_val = X_val.reshape((vdata_size, 3, imsize, imsize))

    network = all_CNN_C(x)

    train_prediction = lasagne.layers.get_output(network)
    train_loss = lasagne.objectives.categorical_crossentropy(train_prediction, y)
    train_loss = train_loss.mean()

    # Regularization
    l2_penalty = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
    train_loss += lambda_decay * l2_penalty

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        train_loss, params, learning_rate=lr, momentum=0.9)

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
    # test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), y),
    #                  dtype=theano.config.floatX)

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
        test_loss,
        givens={
            x: X_test[index * batch_size: (index + 1) * batch_size],
            y: y_test[index * batch_size: (index + 1) * batch_size]
        }
    )

    train_nn(train_fn, val_fn, test_fn,
             n_train_batches, n_valid_batches, n_test_batches, num_epochs,
             verbose=True)


if __name__ == "__main__":

    conv_architecture = ConvPool_CNN_C
    if len(sys.argv) > 1:
        if sys.argv[1] == 'all':
            conv_architecture = all_CNN_C
        elif sys.argv[1] == 'strided':
            conv_architecture = Strided_CNN_C
        elif sys.argv[1] == 'convpool':
            conv_architecture = ConvPool_CNN_C
        else:
            raise NotImplementedError

        run_experiment(lr=0.01, batch_size=200, verbose=True, conv_arch=conv_architecture)

"""
Project NN
"""

from __future__ import print_function
import timeit
import inspect
import sys
import numpy as np
import lasagne
import scipy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample


def ConvPool_CNN_A(input_var=None, filter_size=(3, 3), n_class=10):
    """
    Implementation of the baseline ConvPool-CNN-A model
    :param input_var:
    :param filter_size:
    :param n_class:
    :return:
    """
    imsize = 32

    network = lasagne.layers.InputLayer(
        shape=(None, 3, imsize, imsize),
        stride=(1, 1), pad=1,
        input_var=input_var, filter_size=(3, 3))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=96, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        W=lasagne.init.GlorotUniform(),
        pad=1,
        stride=(1, 1))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3), stride=2, pad=1)
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=192, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        pad=1,
        stride=(1, 1))
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3), stride=2, pad=2)
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=192, filter_size=filter_size,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        stride=(1, 1))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=192, filter_size=(1, 1),
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        stride=(1, 1))
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=10, filter_size=(1, 1),
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        stride=(1, 1))
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Pool2DLayer(network, pool_size=(6, 6), mode='average_inc_pad')
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=n_class,
        nonlinearity=lasagne.nonlinearities.softmax)
    print(lasagne.layers.get_output_shape(network))

    return network


def Strided_CNN_C(input_var=None, filter_size=(3, 3), n_class=10):
    imsize = 32

    network = lasagne.layers.InputLayer(
        shape=(None, 3, imsize, imsize),
        stride=(1, 1), pad=1,
        input_var=input_var)

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        lasagne.layers.DropoutLayer(network, p=0.2), num_filters=96, filter_size=filter_size,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        W=lasagne.init.GlorotUniform(),
        pad=1,
        stride=(1, 1))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=96, filter_size=filter_size,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        pad=1,
        stride=(2, 2))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        lasagne.layers.DropoutLayer(network, p=0.5),
        num_filters=192, filter_size=filter_size,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        pad=1,
        stride=(1, 1))
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=192, filter_size=filter_size,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        pad=1,
        stride=(2, 2))
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        lasagne.layers.DropoutLayer(network, p=0.5),
        num_filters=192, filter_size=filter_size,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        stride=(1, 1))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=192, filter_size=(1, 1),
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        stride=(1, 1))
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=10, filter_size=(1, 1),
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        stride=(1, 1))
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Pool2DLayer(network, pool_size=(6, 6), mode='average_inc_pad')
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=n_class,
        nonlinearity=lasagne.nonlinearities.softmax)
    print(lasagne.layers.get_output_shape(network))

    return network


def ConvPool_CNN_C(input_var=None, filter_size=(3, 3), n_class=10):
    imsize = 32

    network = lasagne.layers.InputLayer(
        shape=(None, 3, imsize, imsize),
        stride=(1, 1), pad=1,
        input_var=input_var)

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        lasagne.layers.DropoutLayer(network, p=0.2), num_filters=96, filter_size=filter_size,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        W=lasagne.init.GlorotUniform(),
        pad=1,
        stride=(1, 1))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=96, filter_size=filter_size,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        pad=1,
        stride=(1, 1))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=96, filter_size=filter_size,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        pad=1,
        stride=(1, 1))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.MaxPool2DLayer(
        network,
        pool_size=(3, 3),
        stride=2,
        pad=1)

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        lasagne.layers.DropoutLayer(network, p=0.5),
        num_filters=192,
        filter_size=filter_size,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        pad=1,
        stride=(1, 1))
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=192, filter_size=filter_size,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        pad=1,
        stride=(1, 1))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=192, filter_size=filter_size,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        pad=1,
        stride=(1, 1))
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3), stride=2, pad=1)
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        lasagne.layers.DropoutLayer(network, p=0.5),
        num_filters=192,
        filter_size=filter_size,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        stride=(1, 1))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=192, filter_size=(1, 1),
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        stride=(1, 1))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=10, filter_size=(1, 1),
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        stride=(1, 1))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Pool2DLayer(
        network,
        pool_size=(6, 6),
        mode='average_inc_pad')

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=n_class,
        nonlinearity=lasagne.nonlinearities.softmax)
    print(lasagne.layers.get_output_shape(network))

    return network


def all_CNN_C(input_var=None, filter_size=(3, 3), n_class=10):
    imsize = 32

    network = lasagne.layers.InputLayer(shape=(None, 3, imsize, imsize), stride=(1, 1), pad=1, input_var=input_var)

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        lasagne.layers.DropoutLayer(network, p=0.2),
        num_filters=96, filter_size=filter_size,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        W=lasagne.init.GlorotUniform(),
        pad=1,
        stride=(1, 1))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=96, filter_size=filter_size,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        pad=1,
        stride=(1, 1))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=96, filter_size=filter_size,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        pad=1,
        stride=(2, 2))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        lasagne.layers.DropoutLayer(network, p=0.5),
        num_filters=192, filter_size=filter_size,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        pad=1,
        stride=(1, 1))
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=192, filter_size=filter_size,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        pad=1,
        stride=(1, 1))
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=192, filter_size=filter_size,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        pad=1,
        stride=(2, 2))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        lasagne.layers.DropoutLayer(network, p=0.2),
        num_filters=192, filter_size=filter_size,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        stride=(1, 1))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=192, filter_size=(1, 1),
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        stride=(1, 1))
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=10, filter_size=(1, 1),
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        stride=(1, 1))
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Pool2DLayer(network, pool_size=(6, 6), mode='average_inc_pad')
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=n_class,
        nonlinearity=lasagne.nonlinearities.softmax)
    print(lasagne.layers.get_output_shape(network))
    return network


def train_nn(train_model, validate_model, test_model,
             n_train_batches, n_valid_batches, n_test_batches, n_epochs,
             verbose=True):
    """
    Wrapper function for training and test THEANO model

    :type train_model: Theano.function
    :param train_model:

    :type validate_model: Theano.function
    :param validate_model:

    :type test_model: Theano.function
    :param test_model:

    :type n_train_batches: int
    :param n_train_batches: number of training batches

    :type n_valid_batches: int
    :param n_valid_batches: number of validation batches

    :type n_test_batches: int
    :param n_test_batches: number of testing batches

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to

    """

    # early-stopping parameters
    patience = 1000000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.85  # a relative improvement of this much is
    # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
    # go through this many
    # minibatche before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                if verbose:
                    print('epoch %i, minibatch %i/%i, validation error (loss) %f %%' %
                          (epoch,
                           minibatch_index + 1,
                           n_train_batches,
                           this_validation_loss * 100.))

                # if we got the best validation score until now

                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                        ]
                    test_score = np.mean(test_losses[0])
                    # test_accuracy = np.mean(test_losses[1])
                    if verbose:
                        print(('    epoch %i, minibatch %i/%i, test error (loss) of '
                               'best model %f %%.') %
                              (epoch, minibatch_index + 1,
                               n_train_batches,
                               test_score * 100))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    # Print out summary
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)


def errors(y_pred, y):
    y_pred = theano.tensor.argmax(y_pred, axis=-1)

    if y.ndim != y_pred.ndim:
        raise TypeError(
            'y should have the same shape as y_pred',
            ('y', y.type, 'y_pred', y_pred.type)
        )

    if y.dtype.startswith('int'):
        # the T.neq operator returns a vector of 0s and 1s, where 1
        # represents a mistake in prediction
        return T.mean(T.neq(y_pred, y))



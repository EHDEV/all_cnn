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

def relu(x, alpha=0):

    """
    Compute the element-wise rectified linear activation function.
    .. versionadded:: 0.7.1
    Parameters
    ----------
    x : symbolic tensor
        Tensor to compute the activation function for.
    alpha : scalar or tensor, optional
        Slope for negative input, usually between 0 and 1. The default value
        of 0 will lead to the standard rectifier, 1 will lead to
        a linear activation function, and any value in between will give a
        leaky rectifier. A shared variable (broadcastable against `x`) will
        result in a parameterized rectifier with learnable slope(s).
    Returns
    -------
    symbolic tensor
        Element-wise rectifier applied to `x`.
    Notes
    -----
    This is numerically equivalent to ``T.switch(x > 0, x, alpha * x)``
    (or ``T.maximum(x, alpha * x)`` for ``alpha < 1``), but uses a faster
    formulation or an optimized Op, so we encourage to use this function.
    """
    # This is probably the fastest implementation for GPUs. Both the forward
    # pass and the gradient get compiled into a single GpuElemwise call.
    # TODO: Check if it's optimal for CPU as well; add an "if" clause if not.
    # TODO: Check if there's a faster way for the gradient; create an Op if so.
    if alpha == 0:
        return 0.5 * (x + abs(x))
    else:
        f1 = 0.5 * (1 + alpha)
        f2 = 0.5 * (1 - alpha)
        return f1 * x + f2 * abs(x)

        
def Strided_CNN_C(input_var=None):

    imsize = 32

    network = lasagne.layers.InputLayer(
        shape=(None, 3, imsize, imsize),
        stride=(1,1), pad=1,
        input_var=input_var)

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
                lasagne.layers.DropoutLayer(network, p=0.2), num_filters=96, filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                W=lasagne.init.GlorotUniform(),
                pad=1,
                stride=(1,1))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
                network, num_filters=96, filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                pad=1,
                stride=(2,2))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
                lasagne.layers.DropoutLayer(network, p=0.5),
                num_filters=192, filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                pad=1,
                stride=(1,1))
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
                network, num_filters=192, filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                pad=1,
                stride=(2,2))
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
                lasagne.layers.DropoutLayer(network, p=0.5),
                num_filters=192, filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                stride=(1,1))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
                network, num_filters=192, filter_size=(1, 1),
                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                stride=(1,1))
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
                network, num_filters=10, filter_size=(1, 1),
                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                stride=(1,1))
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Pool2DLayer(network, pool_size=(6, 6), mode='average_inc_pad')
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.DenseLayer(
                lasagne.layers.dropout(network, p=.5),
                num_units=10,
                nonlinearity=lasagne.nonlinearities.softmax)
    print(lasagne.layers.get_output_shape(network))

    return network

def ConvPool_CNN_C(input_var=None):

    imsize = 32
    
    network = lasagne.layers.InputLayer(
        shape=(None, 3, imsize, imsize),
        stride=(1,1), pad=1,
        input_var=input_var)

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
                lasagne.layers.DropoutLayer(network, p=0.2), num_filters=96, filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                W=lasagne.init.GlorotUniform(),
                pad=1,
                stride=(1,1))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
                network, num_filters=96, filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                pad=1,
                stride=(1,1))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
                network, num_filters=96, filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                pad=1,
                stride=(1,1))

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
                filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                pad=1,
                stride=(1,1))
    print(lasagne.layers.get_output_shape(network))


    network = lasagne.layers.Conv2DLayer(
                network, num_filters=192, filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                pad=1,
                stride=(1,1))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
                network, num_filters=192, filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                pad=1,
                stride=(1,1))
    print(lasagne.layers.get_output_shape(network))
    
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3), stride=2,pad=1)
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
                lasagne.layers.DropoutLayer(network, p=0.5),
                num_filters=192,
                filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                stride=(1,1))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
                network, num_filters=192, filter_size=(1, 1),
                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                stride=(1,1))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
                network, num_filters=10, filter_size=(1, 1),
                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                stride=(1,1))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Pool2DLayer(
        network,
        pool_size=(6, 6),
        mode='average_inc_pad')

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.DenseLayer(
                lasagne.layers.dropout(network, p=.5),
                num_units=10,
                nonlinearity=lasagne.nonlinearities.softmax)
    print(lasagne.layers.get_output_shape(network))

    return network

def all_CNN_C(input_var=None):

    imsize = 32
    
    network = lasagne.layers.InputLayer(shape=(None, 3, imsize, imsize), stride=(1,1), pad=1,input_var=input_var)

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
                lasagne.layers.DropoutLayer(network, p=0.2),
                num_filters=96, filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                W=lasagne.init.GlorotUniform(),
                pad=1,
                stride=(1,1))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
                network, num_filters=96, filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                pad=1,
                stride=(1,1))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
                network, num_filters=96, filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                pad=1,
                stride=(2,2))

    print(lasagne.layers.get_output_shape(network))


    network = lasagne.layers.Conv2DLayer(
                lasagne.layers.DropoutLayer(network, p=0.5),
                num_filters=192, filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                pad=1,
                stride=(1,1))
    print(lasagne.layers.get_output_shape(network))


    network = lasagne.layers.Conv2DLayer(
                network, num_filters=192, filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                pad=1,
                stride=(1,1))
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
                network, num_filters=192, filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                pad=1,
                stride=(2,2))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
                lasagne.layers.DropoutLayer(network, p=0.2),
                num_filters=192, filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                stride=(1,1))

    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
                network, num_filters=192, filter_size=(1, 1),
                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                stride=(1,1))
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Conv2DLayer(
                network, num_filters=10, filter_size=(1, 1),
                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                stride=(1,1))
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.Pool2DLayer(network, pool_size=(6, 6), mode='average_inc_pad')
    print(lasagne.layers.get_output_shape(network))

    network = lasagne.layers.DenseLayer(
                lasagne.layers.dropout(network, p=.5),
                num_units=10,
                nonlinearity=lasagne.nonlinearities.softmax)
    print(lasagne.layers.get_output_shape(network))
    return network

def train_nn(train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,
            verbose = True):
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
    patience = 10000  # look as this many examples regardless
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

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
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
                    #test_accuracy = np.mean(test_losses[1])
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

def categorical_accuracy(predictions, targets, top_k=1):
    """Computes the categorical accuracy between predictions and targets.
    .. math:: L_i = \\mathbb{I}(t_i = \\operatorname{argmax}_c p_{i,c})
    Can be relaxed to allow matches among the top :math:`k` predictions:
    .. math::
        L_i = \\mathbb{I}(t_i \\in \\operatorname{argsort}_c (-p_{i,c})_{:k})
    Parameters
    ----------
    predictions : Theano 2D tensor
        Predictions in (0, 1), such as softmax output of a neural network,
        with data points in rows and class probabilities in columns.
    targets : Theano 2D tensor or 1D tensor
        Either a vector of int giving the correct class index per data point
        or a 2D tensor of 1 hot encoding of the correct class in the same
        layout as predictions
    top_k : int
        Regard a prediction to be correct if the target class is among the
        `top_k` largest class probabilities. For the default value of 1, a
        prediction is correct only if the target class is the most probable.
    Returns
    -------
    Theano 1D tensor
        An expression for the item-wise categorical accuracy in {0, 1}
    Notes
    -----
    This is a strictly non differential function as it includes an argmax.
    This objective function should never be used with a gradient calculation.
    It is intended as a convenience for validation and testing not training.
    To obtain the average accuracy, call :func:`theano.tensor.mean()` on the
    result, passing ``dtype=theano.config.floatX`` to compute the mean on GPU.
    """
    if targets.ndim == predictions.ndim:
        targets = theano.tensor.argmax(targets, axis=-1)
    elif targets.ndim != predictions.ndim - 1:
        raise TypeError('rank mismatch between targets and predictions')
    if top_k == 1:
        # standard categorical accuracy
        top = theano.tensor.argmax(predictions, axis=-1)
        return theano.tensor.neq(top, targets)
    else:
        # top-k accuracy
        top = theano.tensor.argsort(predictions, axis=-1)
        # (Theano cannot index with [..., -top_k:], we need to simulate that)
        top = top[[slice(None) for _ in range(top.ndim - 1)] +
                  [slice(-top_k, None)]]
        targets = theano.tensor.shape_padaxis(targets, axis=-1)
        return theano.tensor.any(theano.tensor.eq(top, targets), axis=-1)
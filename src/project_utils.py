"""
Project Utils
"""

import os
import sys
import tarfile
import gzip
import numpy
import scipy.io

import theano
import theano.tensor as T

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def load_data(simple=True, theano_shared=True):
    '''
    Load the ATIS dataset

    :type foldnum: int
    :param foldnum: fold number of the ATIS dataset, ranging from 0 to 4.

    '''
    cifar_url = 'https://www.cs.toronto.edu/~kriz/'
    if simple:
        filename = 'cifar-10-matlab.tar.gz'
        foldname = '/cifar-10-batches-mat'
    else:
        filename = 'cifar-100-matlab.tar.gz'
        foldname = '/cifar-100-batches-mat'

    if not os.path.exists(os.path.realpath("../../data")):
        os.makedirs(os.path.realpath("../../data"))

    datapath = os.path.realpath("../../data")
    # datapath = os.path.join(os.path.split(__file__)[0],"../../","data")
    
    def check_dataset(dataset):
        # Check if dataset is in the data directory.
        import pdb
        pdb.set_trace()
        new_path = os.path.join(os.path.realpath("../../"),
            "data",
            dataset
        )
        if (not os.path.isfile(new_path)):
            from six.moves import urllib
            origin = (cifar_url + dataset)
            print('Downloading data from %s' % origin)
            #import pdb 
            #pdb.set_trace()
            urllib.request.urlretrieve(origin, new_path)
            tfile = tarfile.open(filename, 'r:gz')
            tfile.extractall(datapath)
            tfile.close()
        return new_path

    filename = check_dataset(filename)
    #tfile = tarfile.open(filename, 'r:gz')
    #tfile.extractall(datapath)
    #tfile.close()
    
    #import pdb
    #pdb.set_trace()
    batch_1 = 'data_batch_1.mat'
    batch_2 = 'data_batch_2.mat'
    batch_3 = 'data_batch_3.mat'
    batch_4 = 'data_batch_4.mat'
    batch_5 = 'data_batch_5.mat'
    batch_test = 'test_batch.mat'
    
    b1 = scipy.io.loadmat(datapath + foldname + '/' +  batch_1)
    b2 = scipy.io.loadmat(datapath + foldname + '/' + batch_2)
    b3 = scipy.io.loadmat(datapath + foldname + '/' + batch_3)
    b4 = scipy.io.loadmat(datapath + foldname + '/' + batch_4)
    b5 = scipy.io.loadmat(datapath + foldname + '/' + batch_5)
    bt = scipy.io.loadmat(datapath + foldname + '/' + batch_test)
    
    def convert_data_format(data):
        X = data['data'] / 255.
        y = data['labels'].flatten()
        return (X, y) 
    
    b1, b1_l = convert_data_format(b1)
    b2, b2_l = convert_data_format(b2)
    b3, b3_l = convert_data_format(b3)
    b4, b4_l = convert_data_format(b4)
    b5, b5_l = convert_data_format(b5)
    bt, bt_l = convert_data_format(bt)
    
    btrain = numpy.concatenate((b1,b2,b3,b4,b5), axis=0)
    btrain_labels = numpy.concatenate((b1_l,b2_l,b3_l,b4_l,b5_l), axis=0)
    
    train_set = (btrain, btrain_labels)
    test_set = (bt, bt_l)
    
    train_set_len = len(train_set[1])
    
    # Extract validation dataset from train dataset
    valid_set = [x[-(train_set_len//10):] for x in train_set]
    train_set = [x[:-(train_set_len//10)] for x in train_set]
    
    if theano_shared:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    else:
        rval = [train_set, valid_set, test_set]
        
    #try:
    #    train_set, valid_set, test_set, dicts = pickle.load(f, encoding='latin1')
    #except:
    #    train_set, valid_set, test_set, dicts = pickle.load(f)
    return rval

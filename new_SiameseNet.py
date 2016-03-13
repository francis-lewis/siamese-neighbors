from __future__ import absolute_import
#from __future__ import print_function
import numpy as np

import random

import keras as k
from keras.datasets import cifar10
from keras.models import Sequential, Graph
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop
from keras import backend as K

from data_utils import *
from collections import defaultdict

import random

do_save = False
do_load = False

#########################
### Utility Functions ###
#########################

def chopra_loss(y_true, y_pred):
    ''' (1-Y)(2/Q)(Ew)^2 + (Y) 2 Q e^(-2.77/Q * Ew)
        Needs to use functions of keras.backend.theano_backend = K '''
    margin = 1
    loss = K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
    return loss

def l2dist(x):
    assert len(x) == 2
    y, z = x.values()
    return K.sqrt(K.sum(K.square(y - z), axis=1, keepdims=True))

def generate_data(x, d):
    ''' Basically from the example Keras siamese network code. '''
    pairs = []
    labels = []
    num_labels = len(d)
    n = min([len(d[j]) for j in range(num_labels)]) - 1
    for j in range(num_labels):
        for i in range(n):
            z1, z2 = d[j][i], d[j][i+1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_labels)
            jn = (j + inc) % num_labels
            z1, z2 = d[j][i], d[jn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    p = np.array(pairs)
    return [p[:,0], p[:,1]], np.array(labels)

def invert_dataset(x, y):
    d = defaultdict(lambda : [])
    for i, label in enumerate(y):
        d[label].append(i)
    return d

def compute_accuracy(preds, labels):
    return labels[preds.ravel() < 0.5].mean()

##########################
### Siamese Net Object ###
##########################

class SiameseNet:
    ''' A rough implementation of Chopra et al. 2005's Siamese network.
    Essentially a wrapper of a Sequential which takes inputs from a Siamese portion
    and adds one more layer that merges the two identical legs (with a custom merge function). '''

    # Defaults
    TRAINING_BATCH_SIZE   = 64
    TRAINING_NB_EPOCHS    = 30
    VALIDATION_BATCH_SIZE = 1
    PREDICT_BATCH_SIZE    = 1
    
    INPUT_LEFT = 'left'
    INPUT_RIGHT = 'right'
    OUTPUT = 'output'

    def __init__(self, structure, input_shape, verbose=True):
        
        self.input_shape=(3, 32, 32)
        self.verbose = verbose
        self.construct(structure)
        
    def construct(self, structure):
        ''' structure - a list of (is_shared, layer_fn) tuples detailing the structure
                 of the Siamese part of the network
                 is_shared - boolean, whether or not the layer is shared
                 layer_fn - a generator function for a layer '''
        
        self.graph = Graph()
        input_left = self.INPUT_LEFT
        input_right = self.INPUT_RIGHT
        self.graph.add_input(name=input_left, input_shape=self.input_shape)
        self.graph.add_input(name=input_right, input_shape=self.input_shape)
        unique_name = 'name'
        shared_name = 'shared'
        dist_name = 'dist'
        self.graph.add_shared_node(structure, name=shared_name, 
            inputs=[input_left, input_right], merge_mode='join') 
        self.graph.add_node(Lambda(l2dist),
                input=shared_name,
                name=dist_name)
        self.graph.add_output(name=self.OUTPUT, input=dist_name)
        if self.verbose:
            print 'Constructed a SiameseNet.'
    
    def compile(self):
        #sgd = SGD(lr=1e-7, decay=1e-6, momentum=0.9, nesterov=True)
        self.graph.compile(loss={'output': chopra_loss}, optimizer='adam')
        if self.verbose:
            print 'Successfully compiled the SiameseNet.'
            
    def _transform_data(self, x, y=None):
        data = {
                self.INPUT_LEFT: x[0],
                self.INPUT_RIGHT: x[1]
            }
        if y is not None:
            data[self.OUTPUT] = y
        return data
        
    def fit(self, x, y, validation_data=None, nb_epoch=TRAINING_NB_EPOCHS,
            batch_size=TRAINING_BATCH_SIZE, shuffle=True):
        ''' Train it. '''
        if validation_data is not None:
            validation_data = self._transform_data(validation_data[0], validation_data[1])
        history = self.graph.fit(self._transform_data(x, y), validation_data=validation_data, nb_epoch=nb_epoch, batch_size=batch_size)
        if self.verbose:
            print 'Done training the SiameseNet.'
        return history
        
    def evaluate(self, x, y, batch_size=VALIDATION_BATCH_SIZE):
        ''' Validate it. '''
        validation_loss = self.graph.evaluate(self._transform_data(x, y), batch_size=batch_size)
        if self.verbose:
            print 'Validation loss is', validation_loss
        return validation_loss
        
    def predict(self, x, batch_size=PREDICT_BATCH_SIZE):
        ''' Predict it. (Not sure if this is helpful) '''
        prediction = self.graph.predict(self._transform_data(x), batch_size=batch_size)
        if self.verbose:
            print 'Predicted probabilities are', prediction
        return prediction
        
    def save(self, filepath):
        print 'Saving weights...'
        self.graph.save_weights(filepath)
        print 'Done saving the weights.'
        
    def load(self, filepath):
        self.graph.load_weights(filepath)
        
    """
    def similarity(self, x1, x2):
        x = [x1, x2]
        prediction = self.graph.predict(self._transform_data(x), batch_size=1)
        return prediction['output']
    """
    
############
### Main ###
############

"""
def _train_sn(sn, x_train, y_train, filepath):
    d_train = invert_dataset(x_train,  y_train)
    history = sn.fit(*generate_data(x_train, d_train))
    if do_save:
        sn.save(filepath)
    return history
"""
def main():

    # Prepare data
    print 'Getting CIFAR10 data...'
    
    data = get_CIFAR10_data()

    x_train, y_train = data['X_train'], data['y_train']
    x_val,   y_val   = data['X_val'],   data['y_val']
    
    N = x_train.shape[0]
    
    # Specify structure of Siamese part of SiameseNet
    # This part needs to be improved. I'm kind of just using random layers.
    init = 'glorot_uniform'
    in_shp = (3,32,32)
    seq = Sequential()
    seq.add(BatchNormalization(epsilon=1e-7,
                                mode=0,
                                axis=1,
                                momentum=0.9,
                                weights=None,
                                input_shape=in_shp))
    """
    #layers = []
    layers.append((
            False,
            lambda : BatchNormalization(
                    epsilon=1e-6,
                    mode=0,
                    axis=1,
                    momentum=0.9,
                    weights=None)
            )) # Not-yet-tuned batch norm without shared weights
    layers.append((True, lambda : Convolution2D(10, 3, 3, init=init, border_mode='same')))
    for _ in xrange(1):
        layers.append((True, lambda : Convolution2D(10, 3, 3, init=init, border_mode='same')))
        layers.append((False, lambda : Activation('relu'))) # ReLU activation without shared weights
    layers.append((False, lambda : Flatten()))
    layers.append((False, lambda : Dense(100)))
    """
    seq.add(Flatten())
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    layers = seq

    sn = SiameseNet(layers, input_shape=in_shp, verbose=True)
    sn.compile()
    if do_load:
        sn.load(filepath='weights.h5')
    history = _train_sn(sn, x_train, y_train, filepath='weights.h5')
    print history


    d_val = invert_dataset(x_val,  y_val)
    loss = sn.evaluate(*generate_data(x_val, d_val))

    val_x_dat, val_y_dat = generate_data(x_val, d_val)
    prediction = sn.predict(val_x_dat)[SiameseNet.OUTPUT]

    ret_preds = prediction
    max_d = np.max(ret_preds)
    min_d = np.min(ret_preds)
    print max_d
    print min_d
    thresh = (max_d + min_d) / 2.0 
    preds = [0,0]
    for i,p in enumerate(prediction):
        if ret_preds[i] > thresh:
            preds[1] += 1
        else:
            preds[0] += 1

    print preds

if __name__ == '__main__':
    main()
    

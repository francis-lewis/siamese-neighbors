import keras as k
from keras.models import Sequential
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from data_utils import *
from collections import defaultdict

import random

#########################
### Utility Functions ###
#########################

def chopra_loss(y_true, y_pred):
    ''' (1-Y)(2/Q)(Ew)^2 + (Y) 2 Q e^(-2.77/Q * Ew)
        Needs to use functions of keras.backend.theano_backend = K '''
    Q = 100.
    return (1 - y_true) * 2 / Q * K.square(y_pred) + y_true * 2 * Q * K.exp(-2.77 / Q * y_pred)

def l2dist(x):
    ''' Chopra '05 computes output = || G(X_1) - G(X_2) ||
        x[0] is G(X_1)
        x[1] is G(X_2) '''
    #return K.sqrt(K.sum(K.square(x[0] - x[1]), axis=1, keepdims=True))
    return K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True)
    
def generate_data(d, examples_per_image=1):
    ''' Generates 50% genuine and 50% impostor pairs
        Returns a ([left_x, right_x], y_target) tuple. '''
    print 'Generating data...'
    (x_genuine_1, x_genuine_2), y_genuine = generate_genuine_data(d, examples_per_image=examples_per_image)
    (x_impostor_1, x_impostor_2), y_impostor = generate_impostor_data(d, examples_per_image=examples_per_image)
    index_permutation = np.random.permutation(np.arange(x_genuine_1.shape[0] + x_impostor_1.shape[0]))
    left_x = np.concatenate((x_genuine_1, x_impostor_1), axis=0)[index_permutation,:]
    right_x = np.concatenate((x_genuine_2, x_impostor_2), axis=0)[index_permutation,:]
    y_target = np.concatenate((y_genuine, y_impostor), axis=0)[index_permutation]
    print 'Done generating data'
    return [left_x, right_x], y_target

def generate_genuine_data(d, examples_per_image=1):
    left_x, right_x = [], []
    for label in d:
        images = d[label]
        num_images = len(images)
        for i in xrange(num_images):
            for j in xrange(examples_per_image): # every image will have examples_per_image genuine matches
                left_x.append(images[i])
                right_x.append(images[random.randint(0, num_images - 1)])
    return [np.array(left_x), np.array(right_x)], np.zeros(len(left_x))
    
def generate_impostor_data(d, examples_per_image=1):
    left_x, right_x = [], []
    for label in d:
        images = d[label]
        num_images = len(images)
        different_labels = [z for z in xrange(len(d)) if z != label]
        for i in xrange(num_images):
            for j in xrange(examples_per_image):
                left_x.append(images[i])
                right_x.append(random.choice(d[random.choice(different_labels)]))
    return [np.array(left_x), np.array(right_x)], np.ones(len(left_x))

def invert_dataset(x, y):
    d = defaultdict(lambda : [])
    for i, label in enumerate(y):
        d[label].append(x[i,:,:,:])
    return d

##########################
### Siamese Net Object ###
##########################

class SiameseNet:
    ''' A rough implementation of Chopra et al. 2005's Siamese network.
    Essentially a wrapper of a Sequential which takes inputs from a Siamese portion
    and adds one more layer that merges the two identical legs (with a custom merge function). '''

    # Defaults
    TRAINING_BATCH_SIZE   = 64
    TRAINING_NB_EPOCHS    = 2
    VALIDATION_BATCH_SIZE = 1
    PREDICT_BATCH_SIZE    = 1

    def __init__(self, structure, verbose=True):
        
        self.verbose = verbose
        self.construct(structure)
        
    def construct(self, structure):
        ''' structure - a list of (is_shared, layer_fn) tuples detailing the structure
                 of the Siamese part of the network
                 is_shared - boolean, whether or not the layer is shared
                 layer_fn - a generator function for a layer '''
        
        inputs = [Sequential(), Sequential()]
        for is_shared, layer_fn in structure:
            if is_shared:
                add_shared_layer(layer_fn(), inputs)
            else:
                for i in xrange(2):
                    inputs[i].add(layer_fn())
        
        self.model = Sequential()
        self.model.add(LambdaMerge(inputs, function=l2dist))
        if self.verbose:
            print 'Constructed a SiameseNet.'
    
    def compile(self):
        self.model.compile(loss=chopra_loss, optimizer='adam')
        #self.model.compile(loss="categorical_crossentropy", optimizer='adam')
        if self.verbose:
            print 'Successfully compiled the SiameseNet.'
        
    def fit(self, x, y, validation_data=None, nb_epoch=TRAINING_NB_EPOCHS,
            batch_size=TRAINING_BATCH_SIZE, shuffle=True):
        ''' Train it. '''
        self.model.fit(x, y, nb_epoch=nb_epoch, batch_size=batch_size)
        if self.verbose:
            print 'Done training the SiameseNet.'
        
    def evaluate(self, x, y, batch_size=VALIDATION_BATCH_SIZE):
        ''' Validate it. '''
        validation_loss = self.model.evaluate(x, y, batch_size=batch_size)
        if self.verbose:
            print 'Validation loss is', validation_loss
        return validation_loss
        
    def predict(self, x, batch_size=PREDICT_BATCH_SIZE):
        ''' Predict it. (Not sure if this is helpful) '''
        prediction = self.model.predict(x, batch_size=batch_size)
        if self.verbose:
            print 'Predicted probabilities are', prediction
        return prediction
        
    def similarity(self, x1, x2):
        pass # The crux of this project
    
############
### Main ###
############

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
    layers = []
    layers.append((
            False,
            lambda : BatchNormalization(
                    epsilon=1e-6,
                    mode=0,
                    axis=1,
                    momentum=0.9,
                    weights=None,
                    input_shape=(3, 32, 32))
            )) # Not-yet-tuned batch norm without shared weights
    layers.append((True, lambda : Convolution2D(1, 3, 3, init=init, border_mode='same')))
    for _ in xrange(1):
        layers.append((True, lambda : Convolution2D(1, 3, 3, init=init, border_mode='same')))
        layers.append((False, lambda : Activation('relu'))) # ReLU activation without shared weights
    layers.append((False, lambda : Flatten()))
    layers.append((False, lambda : Dense(1)))

    sn = SiameseNet(layers, verbose=True)
    sn.compile()

    d_train = invert_dataset(x_train,  y_train)
    sn.fit(*generate_data(d_train, examples_per_image=1)) #, validation_data=generate_data(x_val, y_val))

    d_val = invert_dataset(x_val,  y_val)
    loss = sn.evaluate(*generate_data(d_val, examples_per_image=5))

    val_x_dat, val_y_dat = generate_data(d_val, examples_per_image=5)
    prediction = sn.predict(val_x_dat)

    preds = [0,0]
    for i,p in enumerate(prediction):
        if val_y_dat[i] > .5:
            preds[1] += p[0]
        else:
            preds[0] += p[0]
    print preds

if __name__ == '__main__':
    main()
    

import keras as k
from keras.models import Sequential
from keras.layers.core import *
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from data_utils import *

#########################
### Utility Functions ###
#########################

def reshape_array(arr):
    ''' Reshapes a N dimensional array into a 2D array '''
    N = arr.shape[0]
    arr = arr.reshape((N, -1))
    return arr

def create_one_hot(y):
    ''' Creates a one-hot vector for the class y '''
    y_hot = np.zeros(10)
    y_hot[y] = 1.0
    return y_hot

def convert_y(arr):
    ''' Converts array of classes into array of one-hot vectors '''
    out = np.zeros((arr.shape[0], 10))
    for i in xrange(arr.shape[0]):
        out[i] = create_one_hot(arr[i])
    return out

def reshape_data(data):
    ''' Reshapes images into a matrix
        and classes vector into a one-hot matrix '''
    temp = data
    for k, v in data.iteritems():
        if 'X' in k:
            temp[k] = reshape_array(v)
        else:
            temp[k] = convert_y(v)
    return temp

def chopra_loss(y_true, y_pred):
    ''' (1-Y)(2/Q)(Ew)^2 + (Y) 2 Q e^(-2.77/Q * Ew)
        Needs to use functions of keras.backend.theano_backend = K '''
    Q = 100.
    return (1 - y_true) * 2 / Q * K.square(y_pred) + y_true * 2 * Q * K.exp(-2.77 / Q * y_pred)

def l2dist(x):
    ''' Chopra '05 computes output = || G(X_1) - G(X_2) ||
        x[0] is G(X_1)
        x[1] is G(X_2) '''
    return K.sqrt(K.sum(K.square(x[0] - x[1]), axis=1, keepdims=True))
    
def generate_data(x, y):
    ''' Generates approximately 55% genuine and 45% impostor pairs
        Returns a ([left_x, right_x], y_target) tuple. '''
    (x_genuine_1, x_genuine_2), y_genuine = generate_genuine_data(x, y)
    (x_impostor_1, x_impostor_2), y_impostor = generate_mostly_impostor_data(x, y)
    index_permutation = np.random.permutation(np.arange(x.shape[0] * 2))
    left_x = np.concatenate((x_genuine_1, x_impostor_1), axis=0)[index_permutation,:]
    right_x = np.concatenate((x_genuine_2, x_impostor_2), axis=0)[index_permutation,:]
    y_target = np.concatenate((y_genuine, y_impostor), axis=0)[index_permutation]
    return [left_x, right_x], y_target

def generate_genuine_data(x, y):
    return [x, x], np.zeros((y.shape[0],))
    
def generate_mostly_impostor_data(x, y):
    r = np.arange(x.shape[0])
    rand_ind = np.random.permutation(r)
    return [x, x[rand_ind,:]], (np.argmax(y, axis=1) != (np.argmax(y, axis=1)[rand_ind]))

##########################
### Siamese Net Object ###
##########################

class SiameseNet:
    ''' A rough implementation of Chopra et al. 2005's Siamese network.
    Essentially a wrapper of a Sequential which takes inputs from a Siamese portion
    and adds one more layer that merges the two identical legs (with a custom merge function). '''

    # Defaults
    TRAINING_BATCH_SIZE   = 64
    TRAINING_NB_EPOCHS    = 5
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
    data = reshape_data(data)

    x_train, y_train = data['X_train'], data['y_train']
    x_val,   y_val   = data['X_val'],   data['y_val']
    
    N = x_train.shape[1]
    
    # Specify structure of Siamese part of SiameseNet
    # This part needs to be improved. I'm kind of just using random layers.
    init = 'glorot_uniform'
    layers = []
    layers.append((
            False,
            lambda : BatchNormalization(
                    epsilon=1e-6,
                    mode=0,
                    axis=-1,
                    momentum=0.9,
                    weights=None,
                    input_shape=(N,))
            )) # Not-yet-tuned batch norm without shared weights
    for _ in xrange(5):
        layers.append((True, lambda : Dense(10, init=init))) # Dense layers with shared weights
        layers.append((False, lambda : Activation('relu'))) # ReLU activation without shared weights

    sn = SiameseNet(layers, verbose=True)
    sn.compile()
    sn.fit(*generate_data(x_train, y_train)) #, validation_data=generate_data(x_val, y_val))
    loss = sn.evaluate(*generate_data(x_val, y_val))
    prediction = sn.predict(generate_data(x_val, y_val)[0])
    
if __name__ == '__main__':
    main()
    
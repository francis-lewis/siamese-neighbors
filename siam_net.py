import keras as k
from data_utils import *
from keras.models import Sequential
from keras.layers.core import *
from keras.layers.normalization import BatchNormalization
from keras import backend as K

def reshapeArray(arr):
    N = arr.shape[0]
    arr = arr.reshape((N, -1))
    return arr

def createOneHot(y):
    y_hot = np.zeros(10)
    y_hot[y] = 1.0
    return y_hot

def convertY(arr):
    out = np.zeros((arr.shape[0],10))
    for i in range(arr.shape[0]):
        out[i] = createOneHot(arr[i])
    return out

def reshapeData(data):
    temp = data
    for k, v in data.iteritems():
        if 'X' in k:
            temp[k] = reshapeArray(v)
        else:
            temp[k] = convertY(v)
    return temp
    
def chopra_loss(y_true, y_pred):
    ''' (1-Y)(2/Q)(Ew)^2 + Y2Qe^(-2.77/Q*Ew) '''
    Q = 100.
    return (1 - y_true) * 2 / Q * K.square(y_pred) + y_true * 2 * Q * K.exp(-2.77 / Q * y_pred)

print 'Getting CIFAR10 data...'
data = get_CIFAR10_data()
data = reshapeData(data)

x_train, y_train, x_val, y_val = data['X_train'], data['y_train'], data['X_val'], data['y_val']

print x_train.shape
print y_train.shape

input_dim = x_train.shape[1]
print 'input_dim:', input_dim

print('Building model...')

left_model = Sequential()
right_model = Sequential()

init = 'glorot_uniform'
inputs = [left_model, right_model]

# Siamese layers can't be the first layer, for some reason (indexing bug?)
# so we add a dummy layer
def identity(x):
    return x
left_model.add(Lambda(identity, input_shape=(input_dim,)))
right_model.add(Lambda(identity, input_shape=(input_dim,)))

# Add 6 dense layers (varying the hidden dimensions gives me errors?)
add_shared_layer(Dense(output_dim=10, input_dim=input_dim, init=init), inputs)
left_model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None))
right_model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None))
left_model.add(Activation('relu'))
right_model.add(Activation('relu'))

for i in range(5):
    add_shared_layer(Dense(10, init=init), inputs)
    left_model.add(Activation('relu'))
    right_model.add(Activation('relu'))

model = Sequential()

# Chopra '05 computes output = || G(X_1) - G(X_2) ||
def l2dist(x):
    ''' x[0] is the output from left
        x[1] is the output from right '''
    z = x[0] - x[1]
    return K.sqrt(K.sum(K.square(z), axis=1, keepdims=True))
    #import theano.tensor as T
    #return T.sqrt((T.sqr(x[0] - x[1])).sum(axis=1, keepdims=True))


model.add(LambdaMerge([left_model, right_model], function=l2dist))
#model.add(Merge([left_model, right_model], mode='mul'))

print 'Compiling model...'
model.compile(loss=chopra_loss, optimizer='adam')

print 'Fitting model...'
#model.fit([x_train, x_train], np.zeros_like(y_train), nb_epoch=5, batch_size=64)

def generate_impostor_data(x, y):
    r = np.arange(x.shape[0])
    rand_ind = np.random.permutation(r)
    return x, x[rand_ind,:], (np.argmax(y, axis=1) != (np.argmax(y, axis=1)[rand_ind]))

left_perm, right_perm, y = generate_impostor_data(x_train, y_train)
y = np.concatenate((np.zeros((y_train.shape[0],)), y), axis=0)
y.reshape(98000, 1)
print y
model.fit([np.concatenate((x_train, left_perm), axis=0), np.concatenate((x_train, right_perm), axis=0)],
        y, nb_epoch=5, batch_size=64)

print 'Validating model...'
val_loss = model.evaluate([x_val, x_val], np.zeros(x_val.shape[0],), batch_size=1)
print 'validation loss: %f' % val_loss
#left_perm, right_perm, y = generate_impostor_data(x_val, y_val)
#preds = model.predict_classes([np.concatenate((x_val, left_perm), axis=0), np.concatenate((x_val, right_perm), axis=0)], batch_size=1)
#preds = model.predict_classes([left_perm, right_perm], batch_size=1)
preds = model.predict_classes([x_train, x_train], batch_size=1)
print preds
print preds.sum()
#acc = [1 for i,p in enumerate(preds) if np.argmax(p) == np.argmax(data['y_val'][i])]
#acc = np.array(acc).sum() / preds.shape[0]
#print('Accuracy: %f' % acc)

import keras as k
from data_utils import *
from keras.models import Sequential
from keras.layers.core import *
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
    # Is something wrong here? loss is nan
    Q = 10.
    return (1 - y_true) * 2 / Q * (y_pred ** 2) + y_true * 2 * Q * K.exp(-2.77 / Q * y_pred)

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
    return K.sqrt(K.sum(K.square(z), keepdims=True))
model.add(LambdaMerge([left_model, right_model], function=l2dist))

print 'Compiling model...'
model.compile(loss=chopra_loss, optimizer='adam')

print 'Fitting model...'
model.fit([x_train, x_train], np.zeros_like(y_train), nb_epoch=5, batch_size=64)

print 'Validating model...'
val_loss = model.evaluate([x_val, x_val], np.zeros_like(y_val), batch_size=1)
print 'validation loss: %f' % val_loss
preds = model.predict_classes([x_val, x_val], batch_size=1)
print preds
#acc = [1 for i,p in enumerate(preds) if np.argmax(p) == np.argmax(data['y_val'][i])]
#acc = np.array(acc).sum() / preds.shape[0]
#print('Accuracy: %f' % acc)

import keras as k
from data_utils import *
from keras.models import Sequential
from keras.layers.core import Dense, Activation

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
        if "X" in k:
            temp[k] = reshapeArray(v)
        else:
            temp[k] = convertY(v)
    return temp

print("Getting CIFAR10 data...")
data = get_CIFAR10_data()
data = reshapeData(data)
x_train = data["X_train"]
y_train = data["y_train"]
print(x_train.shape)
print(y_train.shape)

input_dim = x_train.shape[1]

print("Building model...")
model = Sequential()

init = "glorot_uniform"

model.add(Dense(output_dim=10, input_dim=input_dim, init=init))
model.add(Activation("relu"))
for i in range(5):
    model.add(Dense(output_dim=10, init=init))
    model.add(Activation("relu"))
model.add(Dense(output_dim=10, init=init))
model.add(Activation("softmax"))

print("Compiling model...")
model.compile(loss='categorical_crossentropy', optimizer='adam')

print("Fitting model...")
model.fit(data["X_train"], data["y_train"], nb_epoch=5, batch_size=64)

print("Validating model...")
val_loss = model.evaluate(data["X_val"], data["y_val"], batch_size=1)
print("validation loss: %f" % val_loss)
preds = model.predict_classes(data["X_val"], batch_size=1)
acc = [1 for i,p in enumerate(preds) if np.argmax(p) == np.argmax(data["y_val"][i])]
acc = np.array(acc).sum() / preds.shape[0]
print("Accuracy: %f" % acc)

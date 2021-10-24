import numpy
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.utils import np_utils
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback

def _get_available_gpus():
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
# define baseline model

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(1000, input_dim=num_pixels, init='normal', activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(.57))
	model.add(Dense(num_pixels, init='normal', activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(.3))
	model.add(Dense(num_pixels, init='normal', activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(.57))
	model.add(Dense(256, init='normal', activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(.56))
	model.add(Dense(256, init='normal', activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(.15))
	model.add(Dense(256, init='normal', activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(.72))
	model.add(Dense(256, init='normal', activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(.35))
	model.add(Dense(128, init='normal', activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(.6))
	model.add(Dense(num_classes, init='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# build the model
model = baseline_model()
# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=30, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

# Plot the data
fig, axs = plt.subplots(1, 2)
axs[0].plot(history.history['accuracy'], color='b')
axs[0].plot(history.history['val_accuracy'], color='r')
axs[0].title.set_text('model accuracy')
axs[0].set_ylabel('accuracy')
axs[0].set_xlabel('epoch')
axs[0].legend(['train', 'test'], loc='lower right')

# summarize history for loss
axs[1].plot(history.history['loss'], color='b')
axs[1].plot(history.history['val_loss'], color='r')
axs[1].title.set_text('model loss')
axs[1].set_ylabel('loss')
axs[1].set_xlabel('epoch')
axs[1].legend(['train', 'test'], loc='upper right')

fig.tight_layout(pad=2.0)
plt.savefig("mlp_baseline")
plt.show()
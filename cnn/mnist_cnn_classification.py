from keras import Sequential
import keras
import os
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.datasets import cifar10
from keras.utils import np_utils
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

base_dir = os.getcwd()

# create and configure augmented image generator
datagen_train = ImageDataGenerator(
    width_shift_range=0.1,  # randomly shift images horizontally (10% of total width)
    height_shift_range=0.1,  # randomly shift images vertically (10% of total height)
    horizontal_flip=True)  # randomly flip images horizontally

# use keras to import pre-shuffled CIFAR 10 database
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
"""
image shape: (32, 32, 3)
"""

# rescale [0,255] --> [0,1]
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# break training set into training and validation sets
(X_train, x_valid) = X_train[5000:], X_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

# one-hot encode the labels
num_classes = len(np.unique(y_train))
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

# fit augmented image generator on data
datagen_train.fit(X_train)

# define model
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=X_train[0].shape))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

"""
model.summary()
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 32, 32, 16)        208
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 16, 16, 16)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 16, 16, 32)        2080
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, 8, 32)          0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 8, 8, 64)          8256
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 64)          0
_________________________________________________________________
dropout (Dropout)            (None, 4, 4, 64)          0
_________________________________________________________________
flatten (Flatten)            (None, 1024)              0
_________________________________________________________________
dense (Dense)                (None, 500)               512500
_________________________________________________________________
dropout_1 (Dropout)          (None, 500)               0
_________________________________________________________________
dense_1 (Dense)              (None, 10)                5010
=================================================================
Total params: 528,054
Trainable params: 528,054
Non-trainable params: 0
_________________________________________________________________

"""
# compile the model
opt = keras.optimizers.SGD()
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

batch_size = 32
epochs = 10

# train the model
custom_callbacks = [
    ModelCheckpoint(filepath=base_dir + '/models/mnist.cnn.aug.model.best.hdf5', verbose=1, monitor='val_accuracy',  save_best_only=True)
]
history = model.fit_generator(datagen_train.flow(X_train, y_train, batch_size=batch_size), epochs=epochs,
                              verbose=2, steps_per_epoch=X_train.shape[0] // batch_size,
                              validation_data=(x_valid, y_valid), validation_steps=x_valid.shape[0] // batch_size,
                              callbacks=custom_callbacks)


"""
Optimizer is changed to SGD and accuracy got increased by 11.24 percent
loss: 0.4741 - accuracy: 0.8296 - val_loss: 0.6870 - val_accuracy: 0.7694

"""

from keras import Sequential
import keras
import os
from keras.layers import Dense, Flatten, Dropout
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

base_dir = os.getcwd()

# use keras to import pre-shuffled MNIST database
(X_train, y_train), (X_test, y_test) = mnist.load_data()


def plot_images(no_of_images: int):
    """
    This will display the first n number of images on plot. output of this function is stored in images
    folder with function name
    :param no_of_images: number of images to display
    :return:

    e.g: $ plot_images(6)
    """
    # plot first number of training images
    fig = plt.figure(figsize=(20, 20))
    for i in range(no_of_images):
        ax = fig.add_subplot(1, no_of_images, i + 1, xticks=[], yticks=[])
        ax.imshow(X_train[i], cmap='gray')
        ax.set_title(str(y_train[i]))
    plt.show()


def visualize_input(img):
    """
    This will display the one image on plot. output of this function is stored in images
    folder with function name
    :param img:  send one image as input
    :return:

    e.g: $ visualize_input(X_train[0])
    """

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max() / 2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y], 2)), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y] < thresh else 'black')

    plt.show()


# rescale [0,255] --> [0,1]
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# one-hot encode the labels
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

"""
print('Integer-valued labels:')
print(y_train[:10])
print('One-hot labels:')
print(y_train[:10])

Integer-valued labels:
[5 0 4 1 9 2 1 3 1 4]
One-hot labels:
[[ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.]
 [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.]
 [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
 [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  1.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.]]
"""

# define the model
model = Sequential()
model.add(Flatten(input_shape=X_train[0].shape))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

"""
# summarize the model
model.summary()

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
flatten (Flatten)            (None, 784)               0
_________________________________________________________________
dense (Dense)                (None, 512)               401920
_________________________________________________________________
dropout (Dropout)            (None, 512)               0
_________________________________________________________________
dense_1 (Dense)              (None, 512)               262656
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 10)                5130
=================================================================
Total params: 669,706
Trainable params: 669,706
Non-trainable params: 0
_________________________________________________________________

"""

# compile the model
opt = keras.optimizers.RMSprop(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# train the models with callbacks
"""
since we have limited data that why not using early stopping

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]
"""
custom_callbacks = [
    ModelCheckpoint(filepath=base_dir + '/models/mnist.model.best.hdf5', verbose=1, save_best_only=True)
]

history = model.fit(X_train, y_train, batch_size=128, epochs=20, validation_split=0.2,
                    verbose=1, shuffle=True, callbacks=custom_callbacks)

# load the weights that yielded the best validation accuracy
model.load_weights(base_dir + '/models/mnist.model.best.hdf5')

# evaluate test accuracy
score = model.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]

# print test accuracy
print('Test accuracy: %.4f%%' % accuracy)
"""
Test accuracy: 98.2100%
"""
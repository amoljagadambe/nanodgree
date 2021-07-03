import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.preprocessing.text import Tokenizer

(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=9999,
                                                      skip_top=10,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)

# One-hot encoding the output into vector mode, each of length 9999
tokenizer = Tokenizer(num_words=9999)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

# One-hot encoding the output
num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Reserve 1000 samples for validation
x_val = x_train[-1000:]
y_val = y_train[-1000:]
x_train = x_train[:-1000]
y_train = y_train[:-1000]

# build the model
model = Sequential()
model.add(Dense(528, activation='tanh', input_dim=9999, kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

"""
model.summary():

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 528)               5280000
_________________________________________________________________
dropout (Dropout)            (None, 528)               0
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 1058
=================================================================
Total params: 5,281,058
Trainable params: 5,281,058
Non-trainable params: 0
_________________________________________________________________

"""

# Compiling the model using categorical_crossentropy loss, and rmsprop optimizer.
opt = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# train the model
history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=10,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val, y_val),
)

# Evaluate the model on the test data using `evaluate`
results = model.evaluate(x_test, y_test)
print("test loss: {0}, test accuracy: {1}".format(results[0], results[1]))
"""
test loss: 0.329259991645813, test accuracy: 0.8811200261116028

"""
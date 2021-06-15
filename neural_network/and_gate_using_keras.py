import numpy as np
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Activation


# Set random seed
np.random.seed(42)

# X has shape (num_rows, num_cols), where the training data are stored as row vectors
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
# print(X.shape[1]) --> output: 2

# y must have an output vector for each input vector
y = np.array([[0], [0], [0], [1]], dtype=np.float32)

# One-hot encoding the output
y = np_utils.to_categorical(y)

# Create the Sequential model
and_model = Sequential()

# 1st Layer - Add an input layer of 32 nodes with the same input shape as the training samples in X
and_model.add(Dense(32, input_dim=X.shape[1]))

# Add the softmax activation layer
and_model.add(Activation('softmax'))

# 2nd Layer - Add a fully connected output layer
and_model.add(Dense(2))

# Add a sigmoid activation layer
and_model.add(Activation('sigmoid'))

# Compile the model
and_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# get the summary
and_model.summary()

"""
output:
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 32)                96
_________________________________________________________________
activation (Activation)      (None, 32)                0
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 33
_________________________________________________________________
activation_1 (Activation)    (None, 1)                 0
=================================================================
Total params: 129
Trainable params: 129
Non-trainable params: 0
_________________________________________________________________

"""

# Train the model
and_model.fit(X, y, epochs=1000, verbose=1)

# Evaluate the model
score = and_model.evaluate(X, y)
print("\nAccuracy: ", score[-1])


# Checking the predictions
print("\nPredictions:")
print(and_model.predict_proba(X))

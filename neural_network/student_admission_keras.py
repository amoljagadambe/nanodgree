from keras.utils import np_utils
import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Dense, Activation

# Input and Output data path
cwd = os.getcwd()
data_file_path = os.path.join(cwd, 'data_files/binary.csv')

data = pd.read_csv(data_file_path)

# One-hot encode the rank column & drop the original rank column
data = pd.concat([data, pd.get_dummies(data['rank'], prefix='rank')], axis=1)
data.drop(['rank'], axis=1, inplace=True)

# Normalize the data
data["gre"] = data["gre"] / 800  # max range of the column
data["gpa"] = data["gpa"] / 4  # max gpa on the column

# Separate Features and labels from data frame
X = np.array(data)[:, 1:]

# One hot encode the label
y = np_utils.to_categorical(np.array(data['admit']))

# build the model
model = Sequential()
model.add(Dense(128, input_dim=6))
model.add(Activation('sigmoid'))
model.add(Dense(32))
model.add(Activation('sigmoid'))
model.add(Dense(2))
model.add(Activation('sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

"""
summary output:

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 128)               896
_________________________________________________________________
activation (Activation)      (None, 128)               0
_________________________________________________________________
dense_1 (Dense)              (None, 32)                4128
_________________________________________________________________
activation_1 (Activation)    (None, 32)                0
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 66
_________________________________________________________________
activation_2 (Activation)    (None, 2)                 0
=================================================================
Total params: 5,090
Trainable params: 5,090
Non-trainable params: 0
_________________________________________________________________

"""

# train the model
model.fit(X, y, epochs=1000, batch_size=100, verbose=0)

# evaluate
score = model.evaluate(X, y)

"""
score:

loss: 0.5739012360572815 - accuracy: 0.7049999833106995
"""
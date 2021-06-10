"""
One Hot Encoding

this is used for the categorical values be mapped to integer values
example:
input: 	'red', 'red', 'green'
output: [1, 0]
        [1, 0]
        [0, 1]

STEP 1)
we will use sci-kit learn for this
"""

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.utils import np_utils


# define example
data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']

# change list to np array
values = array(data)
print(values)

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)

# reshape the integer encoded to 2-d array
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)

# invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
print(argmax(onehot_encoded[4, :]))


"""
Output:

values: ['cold' 'cold' 'warm' 'cold' 'hot' 'hot' 'warm' 'cold' 'warm' 'hot']
 
integer_encoded: [0 0 2 0 1 1 2 0 2 1]
 
onehot_encoded:[[ 1.  0.  0.]
                 [ 1.  0.  0.]
                 [ 0.  0.  1.]
                 [ 1.  0.  0.]
                 [ 0.  1.  0.]
                 [ 0.  1.  0.]
                 [ 0.  0.  1.]
                 [ 1.  0.  0.]
                 [ 0.  0.  1.]
                 [ 0.  1.  0.]]
 
inverted:  ['cold']
"""


"""
STEP 2)

Use the keras library
"""

# define example
data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
data = array(data)
print(data)

# convert to integer
integer_encoded = label_encoder.fit_transform(data)

# reshape the integer encoded to 2-d array
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

# one hot encode
encoded = np_utils.to_categorical(data)
print(encoded)

# invert encoding
inverted = argmax(encoded[0])
print(inverted)

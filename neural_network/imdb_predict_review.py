from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import models
import os

BASE_DIR = os.getcwd()

predict_sentiment = [
    "This movie is worst, cast and crew did treble job and honestly storyline is also bad and skeptical. i hate this movie and gives thumbs down"
]

# One-hot encoding the output into vector mode, each of length 9999
tokenizer = Tokenizer(num_words=9999)
tokenizer.fit_on_texts(predict_sentiment)
sentence_encoding = tokenizer.texts_to_matrix(predict_sentiment, mode='binary')
# Pad the training sequences
train_padded = pad_sequences(sentence_encoding,  maxlen=9999)
# It can be used to reconstruct the model identically.
reconstructed_model = models.load_model(BASE_DIR + '/models/imdb_review.h5')

prediction = reconstructed_model.predict(train_padded)
for pred in prediction:
    y_classes = pred.argmax()
    print(y_classes)
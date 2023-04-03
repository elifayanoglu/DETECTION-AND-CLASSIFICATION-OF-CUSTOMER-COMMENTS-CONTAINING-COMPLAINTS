# DataFrame
import pandas as pd

# Matplot
import matplotlib.pyplot as plt

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

# Keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM, SpatialDropout1D
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf

# nltk
import nltk
from nltk.corpus import stopwords

# Word2vec
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors


# Utility
import re    #importing regex module
import numpy as np
import os
from collections import Counter
import logging
import time
import pickle4 as pickle
import itertools
from datasets import load_dataset

# Set log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

path = "toplam_set_103303.csv"
#path = "set_5000.csv"
#path = "test.csv"
#path ="normalized_dataset.csv"
df = pd.read_csv(path)

pd.set_option("display.max_colwidth",None)

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^a-zA-Z0-9ğüşöçıİĞÜŞÖÇ]+"

# WORD2VEC 
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 16
W2V_MIN_COUNT = 1

# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 3
BATCH_SIZE = 256

print("Dataset size:", len(df))

# Stopwords
nltk.download('stopwords')
stop_words = stopwords.words("turkish")

def preprocess(text):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
             tokens.append(token)
    return " ".join(tokens)

df.text = df.text.apply(lambda x: preprocess(x))

documents = [_text.split() for _text in df.text] 
print(documents[1])

# w2v_model = gensim.models.Word2Vec(vector_size=300,
#                                    window=W2V_WINDOW, 
#                                    min_count=W2V_MIN_COUNT, 
#                                    workers=8)

w2v_model = Word2Vec(sentences=documents, vector_size=300, window=W2V_WINDOW, min_count=1, workers=8)
w2v_model.build_vocab(documents)

w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)

w2v_model.wv.most_similar("güzel")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df.text)

vocab_size = len(tokenizer.word_index) + 1
print("Total words", vocab_size)

# x_test = pad_sequences(tokenizer.texts_to_sequences(df.text), maxlen=SEQUENCE_LENGTH)

x_train = pad_sequences(tokenizer.texts_to_sequences(df.text), maxlen=SEQUENCE_LENGTH)

trainset_labels = df.pos
# print(trainset_labels)

embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
print(embedding_matrix.shape)

embedding_layer = tf.keras.layers.Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], 
                                            input_length=SEQUENCE_LENGTH, trainable=False)

sequence_input = tf.keras.layers.Input(shape=(SEQUENCE_LENGTH,), dtype='int32')
embedding_sequences = embedding_layer(sequence_input)
x = tf.keras.layers.SpatialDropout1D(0.2)(embedding_sequences) 
x = tf.keras.layers.Conv1D(32, 3, activation='relu')(x) 
x = tf.keras.layers.LSTM(100, dropout=0.2)(x)
x = tf.keras.layers.Dropout(0.3)(x) 
x = tf.keras.layers.Dense(64, activation='relu')(x) 
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(sequence_input, outputs)

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
              EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]
             

model.fit(x_train[:103203], trainset_labels[:103203],
          batch_size=BATCH_SIZE,
          epochs=3,
          callbacks=callbacks)

from sklearn import model_selection, datasets
from sklearn.tree import DecisionTreeClassifier
import joblib
import pickle


# model.save('w2v_cnn_lstm_2.h5')
model.save_weights(f'weights.h5')



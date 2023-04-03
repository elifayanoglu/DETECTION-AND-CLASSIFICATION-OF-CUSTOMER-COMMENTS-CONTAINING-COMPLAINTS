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
import pickle
import itertools
from datasets import load_dataset

# Set log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

path = "toplam_set_103303.csv"
df = pd.read_csv(path)

pd.set_option("display.max_colwidth",None)

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^a-zA-Z0-9ğüşöçıİĞÜŞÖÇ]+"

# WORD2VEC 
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 16
W2V_MIN_COUNT = 10

# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 3
BATCH_SIZE = 256

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

w2v_model = gensim.models.Word2Vec(vector_size=300,
                                   window=W2V_WINDOW, 
                                   min_count=W2V_MIN_COUNT, 
                                   workers=8)

w2v_model.build_vocab(documents)

w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)

w2v_model.wv.most_similar("güzel")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df.text)

vocab_size = len(tokenizer.word_index) + 1
print("Total words", vocab_size)

x_train = pad_sequences(tokenizer.texts_to_sequences(df.text), maxlen=SEQUENCE_LENGTH)

trainset_labels = df.pos
print(trainset_labels)

embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
print(embedding_matrix.shape)


#CNN-1 başlangıç*********************
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
#CNN ile feature extraction yapılıyor Lstm ise feature'lara göre sınıflandrırıyor

model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
              EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]
            
#CNN-1 başlangıç*********************

import numpy as np
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score, f1_score,classification_report, confusion_matrix,roc_auc_score, recall_score,precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

#CNN-2 başlangıç*********************
folds = KFold(n_splits = 10, shuffle = True, random_state = None)
X, y = x_train,trainset_labels
y = np.array(y)

scores = []
scoresf1 = []
scores_recall =[]
scores_pres =[]
scores_roc=[]
TPs= []
FPs= []
FNs= []
TNs= []

for n_fold, (train_index, valid_index) in enumerate(folds.split(X, y)):
    print('\n Fold '+ str(n_fold+1 ) +
          ' \n\n train ids :' +  str(train_index) +
          ' \n\n validation ids :' +  str(valid_index))
    
    X_train, X_valid = X[train_index], X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]
    
    model.fit(X_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=3,
                    validation_split=0.1,
                    callbacks=callbacks)
    score = model.evaluate(X_valid, y_valid, batch_size=BATCH_SIZE)
    y_pred = model.predict(X_valid)

    y_valid_1d = list(y_valid)
    y_pred_1d = []
    y_pred_1d = tf.cast(tf.round(y_pred), tf.int32).numpy().flatten()

    acc_score = accuracy_score(y_valid_1d, y_pred_1d)
    f1score = f1_score(y_valid, y_pred_1d)
    scorerecall= recall_score(y_valid, y_pred_1d)
    scorepres= precision_score(y_valid, y_pred_1d)
    scoreroc= roc_auc_score(y_valid, y_pred_1d)
    scores.append(acc_score)
    scoresf1.append(f1score)
    scores_recall.append(scorerecall)
    scores_pres.append(scorepres)
    scores_roc.append(scoreroc)
    print('\n Accuracy score for Fold ' +str(n_fold+1) + ' --> ' + str(acc_score)+' ')
    print('\n F1 score for Fold ' +str(n_fold+1) + ' --> ' + str(f1score)+' ')
    print('\n Recall score for Fold ' +str(n_fold+1) + ' --> ' + str( scorerecall)+' ')
    print('\n ROC score for Fold ' +str(n_fold+1) + ' --> ' + str(scoreroc)+' ')
    print('\n Pressicion score for Fold ' +str(n_fold+1) + ' --> ' + str(scorepres)+'\n')

    con =confusion_matrix(y_valid, y_pred_1d)
    TPs.append(con[0][0])
    FPs.append(con[0][1])
    FNs.append(con[1][0])
    TNs.append(con[1][1])
    print("\n Confusion Matrix : ")
    print(con)

print(scores)
print(' CNN+LSTM__Avg. accuracy score :' + str(np.mean(scores)))
print(scoresf1)
print(' CNN+LSTM__Avg. F1 score :' + str(np.mean(scoresf1)))
print(scores_recall)
print(' CNN+LSTM__Avg. Recall score :' + str(np.mean(scores_recall)))
print(scores_roc)
print(' CNN+LSTM__Avg. ROC score :' + str(np.mean(scores_roc)))
print(scores_pres)
print(' CNN+LSTM__Avg. Pressicion score :' + str(np.mean(scores_pres)))

conf_mat  = np.empty((2, 2), float)
conf_mat[0][0]=np.mean(TPs)
conf_mat[0][1]=np.mean(FPs)
conf_mat[1][0]=np.mean(FNs)
conf_mat[1][1]=np.mean(TNs)

table = sns.heatmap(conf_mat/np.sum(conf_mat, axis=1)[:, np.newaxis],
                    annot=True, fmt='.2f', cmap='Blues')
print(conf_mat )
table.set_xlabel('\nPredicted Values')
table.set_ylabel('Actual Values')
plt.show()
#CNN-2 başlangıç*********************


class Sequencer():
    
    def __init__(self,
                 all_words,
                 max_words,
                 seq_len,
                 embedding_matrix
                ):
        
        self.seq_len = seq_len
        self.embed_matrix = embedding_matrix
        """
        temp_vocab = Vocab which has all the unique words
        self.vocab = Our last vocab which has only most used N words.
    
        """
        temp_vocab = list(set(all_words))
        self.vocab = []
        self.word_cnts = {}
        """
        Now we'll create a hash map (dict) which includes words and their occurencies
        """
        for word in temp_vocab:
            # 0 does not have a meaning, you can add the word to the list
            # or something different.
            count = len([0 for w in all_words if w == word])
            self.word_cnts[word] = count
            counts = list(self.word_cnts.values())
            indexes = list(range(len(counts)))
        
        # Now we'll sort counts and while sorting them also will sort indexes.
        # We'll use those indexes to find most used N word.
        cnt = 0
        while cnt + 1 != len(counts):
            cnt = 0
            for i in range(len(counts)-1):
                if counts[i] < counts[i+1]:
                    counts[i+1],counts[i] = counts[i],counts[i+1]
                    indexes[i],indexes[i+1] = indexes[i+1],indexes[i]
                else:
                    cnt += 1
        
        for ind in indexes[:max_words]:
            self.vocab.append(temp_vocab[ind])
                    
    def textToVector(self,text):
        # First we need to split the text into its tokens and learn the length
        # If length is shorter than the max len we'll add some spaces (100D vectors which has only zero values)
        # If it's longer than the max len we'll trim from the end.
        tokens = text.split()
        len_v = len(tokens)-1 if len(tokens) < self.seq_len else self.seq_len-1
        vec = []
        for tok in tokens[:len_v]:
            try:
                vec.append(self.embed_matrix[tok])
            except Exception as E:
                pass
        
        last_pieces = self.seq_len - len(vec)
        for i in range(last_pieces):
            vec.append(np.zeros(300,))
        
        return np.asarray(vec).flatten()

sequencer = Sequencer(all_words = [token for seq in documents for token in seq],
              max_words = 60,
              seq_len = 15,
              embedding_matrix = w2v_model.wv
             )

test_vec = sequencer.textToVector("Bu kıyafet çok dar")
print(test_vec[:10])

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
x_vecs = np.asarray([sequencer.textToVector(" ".join(seq)) for seq in documents])
print(x_vecs.shape)

x_vecs = np.array(x_vecs, dtype=object)

print(x_vecs)

from sklearn.decomposition import PCA
pca_model = PCA(n_components=50)
pca_model.fit(x_vecs)
print("Sum of variance ratios: ",sum(pca_model.explained_variance_ratio_))

x_comps = pca_model.transform(x_vecs)

#SVM
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

train_labels = df.pos
print("train labels",train_labels)

folds = KFold(n_splits = 10, shuffle = True, random_state = None)
X, y = x_comps,train_labels 
y = np.array(y)

scores = []
scoresf1 = []
scores_recall =[]
scores_pres =[]
scores_roc=[]
TPs= []
FPs= []
FNs= []
TNs= []

svm_clf = make_pipeline(StandardScaler(), SVC())

for n_fold, (train_index, valid_index) in enumerate(folds.split(X, y)):
    print('\n Fold '+ str(n_fold+1 ) + 
          ' \n\n train ids :' +  str(train_index) +
          ' \n\n validation ids :' +  str(valid_index))
    
    X_train, X_valid = X[train_index], X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]
    
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_valid)
    
    
    acc_score = accuracy_score(y_valid, y_pred)
    f1score = f1_score(y_valid, y_pred)
    scorerecall= recall_score(y_valid, y_pred)
    scorepres= precision_score(y_valid, y_pred)
    scoreroc= roc_auc_score(y_valid, y_pred)
    scores.append(acc_score)
    scoresf1.append(f1score)
    scores_recall.append(scorerecall)
    scores_pres.append(scorepres)
    scores_roc.append(scoreroc)
    print('\n Accuracy score for Fold ' +str(n_fold+1) + ' --> ' + str(acc_score)+' ')
    print('\n F1 score for Fold ' +str(n_fold+1) + ' --> ' + str(f1score)+' ')
    print('\n Recall score for Fold ' +str(n_fold+1) + ' --> ' + str( scorerecall)+' ')
    print('\n ROC score for Fold ' +str(n_fold+1) + ' --> ' + str(scoreroc)+' ')
    print('\n Pressicion score for Fold ' +str(n_fold+1) + ' --> ' + str(scorepres)+'\n') 

    con =confusion_matrix(y_valid, y_pred)
    TPs.append(con[0][0])
    FPs.append(con[0][1])
    FNs.append(con[1][0])
    TNs.append(con[1][1])
    print("\n Confusion Matrix : ")
    print(con)

print(scores)
print(' SVM__Avg. accuracy score :' + str(np.mean(scores)))
print(scoresf1)
print(' SVM__Avg. F1 score :' + str(np.mean(scoresf1)))
print(scores_recall)
print(' SVM__Avg. Recall score :' + str(np.mean(scores_recall)))
print(scores_roc)
print(' SVM__Avg. ROC score :' + str(np.mean(scores_roc)))
print(scores_pres)
print(' SVM__Avg. Pressicion score :' + str(np.mean(scores_pres)))


conf_mat  = np.empty((2, 2), float)
conf_mat[0][0]=np.mean(TPs)
conf_mat[0][1]=np.mean(FPs)
conf_mat[1][0]=np.mean(FNs)
conf_mat[1][1]=np.mean(TNs)

table = sns.heatmap(conf_mat/np.sum(conf_mat, axis=1)[:, np.newaxis], 
                    annot=True, fmt='.2f', cmap='Blues')
print(conf_mat )
table.set_xlabel('\nPredicted Values')
table.set_ylabel('Actual Values')
plt.show()

#LogRec
from sklearn.linear_model import LogisticRegression
import numpy as np

folds = KFold(n_splits = 10, shuffle = True, random_state = None)
X, y = x_comps,train_labels 
y = np.array(y)

scores = []
scoresf1 = []
scores_recall =[]
scores_pres =[]
scores_roc=[]
TPs= []
FPs= []
FNs= []
TNs= []

lr_clf = LogisticRegression()

for n_fold, (train_index, valid_index) in enumerate(folds.split(X, y)):
    print('\n Fold '+ str(n_fold+1 ) + 
          ' \n\n train ids :' +  str(train_index) +
          ' \n\n validation ids :' +  str(valid_index))
    
    X_train, X_valid = X[train_index], X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]
    
    lr_clf.fit(X_train, y_train)
    y_pred = lr_clf.predict(X_valid)
    
    
    acc_score = accuracy_score(y_valid, y_pred)
    f1score = f1_score(y_valid, y_pred)
    scorerecall= recall_score(y_valid, y_pred)
    scorepres= precision_score(y_valid, y_pred)
    scoreroc= roc_auc_score(y_valid, y_pred)
    scores.append(acc_score)
    scoresf1.append(f1score)
    scores_recall.append(scorerecall)
    scores_pres.append(scorepres)
    scores_roc.append(scoreroc)
    print('\n Accuracy score for Fold ' +str(n_fold+1) + ' --> ' + str(acc_score)+' ')
    print('\n F1 score for Fold ' +str(n_fold+1) + ' --> ' + str(f1score)+' ')
    print('\n Recall score for Fold ' +str(n_fold+1) + ' --> ' + str( scorerecall)+' ')
    print('\n ROC score for Fold ' +str(n_fold+1) + ' --> ' + str(scoreroc)+' ')
    print('\n Pressicion score for Fold ' +str(n_fold+1) + ' --> ' + str(scorepres)+'\n') 

    con =confusion_matrix(y_valid, y_pred)
    TPs.append(con[0][0])
    FPs.append(con[0][1])
    FNs.append(con[1][0])
    TNs.append(con[1][1])
    print("\n Confusion Matrix : ")
    print(con)

print(scores)
print(' LogReg__Avg. accuracy score :' + str(np.mean(scores)))
print(scoresf1)
print(' LogReg__Avg. F1 score :' + str(np.mean(scoresf1)))
print(scores_recall)
print(' LogReg__Avg. Recall score :' + str(np.mean(scores_recall)))
print(scores_roc)
print(' LogReg__Avg. ROC score :' + str(np.mean(scores_roc)))
print(scores_pres)
print(' LogReg__Avg. Pressicion score :' + str(np.mean(scores_pres)))


conf_mat  = np.empty((2, 2), float)
conf_mat[0][0]=np.mean(TPs)
conf_mat[0][1]=np.mean(FPs)
conf_mat[1][0]=np.mean(FNs)
conf_mat[1][1]=np.mean(TNs)

table = sns.heatmap(conf_mat/np.sum(conf_mat, axis=1)[:, np.newaxis], 
                    annot=True, fmt='.2f', cmap='Blues')
print(conf_mat )
table.set_xlabel('\nPredicted Values')
table.set_ylabel('Actual Values')
plt.show()
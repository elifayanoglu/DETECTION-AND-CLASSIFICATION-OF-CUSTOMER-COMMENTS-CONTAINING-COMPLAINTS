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


from w2v_kategorize2_before import w2v_kategorize

# Set log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

path = "toplam_set_103303.csv"
#path = "set_5000.csv"
#path = "son_test.csv"
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

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7) 

THRESHOLD = 0.6

# EXPORT
KERAS_MODEL = "model.h5"
WORD2VEC_MODEL = "model.w2v"
TOKENIZER_MODEL = "tokenizer.pkl"
ENCODER_MODEL = "encoder.pkl"

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

w2v_model = gensim.models.Word2Vec(vector_size=300,
                                   window=W2V_WINDOW, 
                                   min_count=W2V_MIN_COUNT, 
                                   workers=8)

w2v_model.build_vocab(documents)

w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)

w2v_model.wv.most_similar("güzel")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df.text)

vocab_size = len(tokenizer.word_index) + 2
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
              

# model.fit(x_train, trainset_labels,
#           batch_size=BATCH_SIZE,
#           epochs=3,
#           callbacks=callbacks)

model.load_weights("weights.h5")


result= model.predict(x_train[103203:])
print(result)

test_labels = df.pos

from sklearn.metrics import accuracy_score, f1_score,classification_report, confusion_matrix,roc_auc_score, recall_score,precision_score

y_pred_1d = []
y_test_1d = list(test_labels[103203:])
y_pred_1d = model.predict(x_train[103203:], verbose=1, batch_size=8000)
y_pred_1d = tf.cast(tf.round(y_pred_1d), tf.int32).numpy().flatten()


acc_score = accuracy_score(y_test_1d, y_pred_1d)
f1score = f1_score(y_test_1d, y_pred_1d)
scorerecall= recall_score(y_test_1d, y_pred_1d)
scorepres= precision_score(y_test_1d, y_pred_1d)
scoreroc= roc_auc_score(y_test_1d, y_pred_1d)
print("acc:",acc_score)
print("f1:",f1score)
print("recall:",scorerecall)
print("precision:",scorepres)
print("roc:",scoreroc)

# print("result size:", len(result))
# deneme = df.text[103203:]
# print("deneme size:", len(deneme))
# *****************************************************************************************************
path2 = "normalized_dataset-enyeni.csv"
# test = "test_normal_birlesti.csv"
df2 = pd.read_csv(path2)

df2.text = df2.text.apply(lambda x: preprocess(x))

documents2 = [_text.split() for _text in df2.text] 

w2v = Word2Vec(documents2, vector_size=300, window=W2V_WINDOW, min_count=1, workers=8)

w2v.build_vocab(documents2)

w2v.train(documents2, total_examples=len(documents2), epochs=W2V_EPOCH)

tokenizer2= Tokenizer()
tokenizer2.fit_on_texts(df2.text)

vocab_size2 = len(tokenizer2.word_index) + 2
print("Total words", vocab_size2)

x_train2 = pad_sequences(tokenizer2.texts_to_sequences(df2.text), maxlen=SEQUENCE_LENGTH)
print(x_train2.shape)


kumas_dikis = ["ince","incecik","inceydi","yırtık","delik","çekti","yırtıldı","sökük","kalitesi","kaliteli","kalitesiz","gösteriyor","naylon","kalın","yamuk","dandik",
"dikiş","dikişleri","dikişi","dikmişler","defolu","terleten","terletiyor","terletecek","terletir","kumaş","kumaşı","kumaşını","kumaşın","küçüldü","kayboldum"]


kumas_vector = []
for i in range(len(kumas_dikis)):
  string = kumas_dikis[i]
  print(string)
  kumas_vector.append(w2v.wv[string])
  print(kumas_vector[i])

renk = ["rengi","rengini","renginin","soluk","solmuş","soluyor","soldu","canlı"]

renk_vector = []
for i in range(len(renk)):
  string = renk[i]
  print(string)
  renk_vector.append(w2v.wv[string])
  print(renk_vector[i])

kalip_beden = ["bol","boldu","büyük","küçük","dardı","dar","geniş","pot","potluk","kesiminde","kesiminden","oversize","kesim","kesimi","kesimleri","kolları","uzun","kısa",
"boyu","kalıbı","kalıp","kalıbını","beden","bedeni","bedenim","bedene","bedenler","bedenleri"]


kalip_vector = []
for i in range(len(kalip_beden)):
  string = kalip_beden[i]
  print(string)
  kalip_vector.append(w2v.wv[string])
  print(kalip_vector[i])

gorselle_alaka = ["alakası","fotoğraf","fotoğrafta","fotoğraftaki","fotoğraftakinden","fotoğrafla","fotoğraftakiyle","göründüğü",
"görseldeki","görseldekiyle","görselle","görsel","resimdeki","resimdekinden","resimdekiyle","resimde"]

gorsel_vector = []
for i in range(len(gorselle_alaka)):
  string = gorselle_alaka[i]
  print(string)
  gorsel_vector.append(w2v.wv[string])
  print(gorsel_vector[i])

kargo = ["teslimat","yavaş","geç","paketleme","kargo","leke","lekeli","etiketsiz","kusurlu","yanlış","eksik","yerine"]

kargo_vector = []
for i in range(len(kargo)):
  string = kargo[i]
  print(string)
  kargo_vector.append(w2v.wv[string])
  print(kargo_vector[i])



# df_test = pd.read_csv("etiketsiz_200_tek.csv")
# df_test.columns = ["text"]
# testler=df_test.text

from numpy.linalg import norm
categories_names = ["kumas_vector","renk_vector","kalip_vector","gorsel_vector","kargo_vector"]
kategoriler = []
kategoriler.append(kumas_vector)
kategoriler.append(renk_vector)
kategoriler.append(kalip_vector)
kategoriler.append(gorsel_vector)
kategoriler.append(kargo_vector)

kategori_isimleri = ["kumas ve dikiş","renk","beden ve kalıp","görselle alakası yok","kargo ve teslimat"]

def getnorm (liste):#NORMALİZASYON İÇİN
    birinci=""
    ikinci=""
    treshold = 0.6
    #listeyi al
    normlist=[]
    if(liste.count(0)!=5):
      normlist.append(normalizasyon( liste[0],30))
      normlist.append(normalizasyon( liste[1],8))
      normlist.append(normalizasyon( liste[2],29))
      normlist.append(normalizasyon( liste[3],16))
      normlist.append(normalizasyon( liste[4],12))
    print(normlist)
    return normlist

def normalizasyon(x,xmax):
    xmin=0
    son= (x-xmin)/(xmax-xmin)
    return son


def caseler_norm (kategori_list,listex):
    birinci=""
    ikinci=""
    new_list=[]
    new_list = listex.copy()
    listex.sort(reverse=True)
    maxlar1=[]
    maxlar2=[]
    maxlar1ort=[]
    maxlar2ort=[]
    max1=listex[0]
    if(listex.count(0)!=5):
      for i in range(5):
        if(new_list[i] == max1):
          maxlar1.append(i)
          listex[0] = 0

      listex.sort(reverse=True)
      max2=listex[0]    
      for i in range(5):
        if(new_list[i]==max2):
          maxlar2.append(i)

      if(len(maxlar1)==1):
        birinci = kategori_isimleri[maxlar1[0]]
        
        if((max1-max2<0.3 and max2!=0)):
          if(len(maxlar2)==1):
              ikinci = kategori_isimleri[maxlar2[0]]
          else:
            for i in range(len(maxlar2)):
              maxlar2ort.append(np.mean(kategori_list[maxlar2[i]])*0.3 + np.max(kategori_list[maxlar2[i]])*0.7)
            maxlar2ort_new = maxlar2ort.copy()
            maxlar2ort.sort(reverse=True)
            for i in range(len(maxlar2)):
              if(maxlar2ort[0]==maxlar2ort_new[i]):
                ikinci = kategori_isimleri[maxlar2[i]]

      else:
        for i in range(len(maxlar1)):
          maxlar1ort.append(np.mean(kategori_list[maxlar1[i]])*0.3 + np.max(kategori_list[maxlar1[i]])*0.7)
        maxlar1ort_new = maxlar1ort.copy()
        maxlar1ort.sort(reverse=True)
        # print("sort eidlmiş maxlarort",maxlar1ort)
        for i in range(len(maxlar1)):
          if(maxlar1ort[0]==maxlar1ort_new[i]):
            birinci = kategori_isimleri[maxlar1[i]]
            maxlar1ort[0]=0
        maxlar1ort.sort(reverse=True)
        for i in range(len(maxlar1)):
          if(maxlar1ort[0] == maxlar1ort_new[i]):
            ikinci = kategori_isimleri[maxlar1[i]]

    else:
      print("hiçbir kategoriye ait değildir.")
    print("birinci: "+birinci)
    if(len(ikinci)>3):
      print(" ikinci: " +ikinci)




for i in range(99):
    print("test edilen cümle:", df.text[103203 + i])
    print(result[i])
    if(result[i] < 0.5):
        print("bu cümle olumsuzdur")
        print("*******************************************************")
        print("W2V ile kategori etiketleme")
        import normalization_pb2 as z_normalize
        import normalization_pb2_grpc as z_normalize_g
        import grpc
        import pandas as pd
        # path = "toplam_set_103303.csv"
        # df = pd.read_csv(path)

        channel = grpc.insecure_channel('localhost:6789')

        norm_stub = z_normalize_g.NormalizationServiceStub(channel)

        def normalize(i):
            response = norm_stub.Normalize(z_normalize.NormalizationRequest(input=i))
            return response.normalized_input

        sample = normalize(df.text[103203 + i])
        print("normal sentence:", sample)

        sentence = preprocess(sample)

        tokenize = sentence.split()

        # print(len(tokenize))

        tokenize_vector = [] 
        for i in range(len(tokenize)):
            string = tokenize[i]
            tokenize_vector.append(w2v.wv[string])
            # print(tokenize_vector[i])



        deneme = [] 
        kategori_max = []   
        liste = []
        threshold_kat_cnt = []
        for i in range(5):
            # print("-------------------------------------------------------")
            # print("Kategori: ",kategori_isimleri[i])
            deneme = []
            liste = []
            for j in range(len(kategoriler[i])):
                cosine = np.dot(tokenize_vector,kategoriler[i][j])/(norm(tokenize_vector, axis=1)*norm(kategoriler[i][j]))
                deneme.append(max(cosine))
                # print("cosine:",cosine)
                if max(cosine) >= THRESHOLD:
                    liste.append(max(cosine))
            threshold_kat_cnt.append(len(liste))
          # print("liste:",liste)
              
          # print(i+1,". kategori için cosineler:")
            kategori_max.append(deneme)
        print("-------------------------------------------------------")
        print("category counter:",threshold_kat_cnt)
        say = 0
        ks = 0
        ki = 0
        new_list = []
        new_list = threshold_kat_cnt.copy()

        for a in range(5):
            print(a+1,". kategori için benzerlikler: ",kategori_max[a])
        print("test edilen cümle: ",df.text[103203 + i])
        import copy
        listex=copy.deepcopy(threshold_kat_cnt)
        listex2=copy.deepcopy(threshold_kat_cnt)
        # caseler3(kategori_max,threshold_kat_cnt)
        # print("2.DENEME KISMI--------------------------------------")
        # caseler3(kategori_max,listex)
        print("normalizasonlu sonuçlar:")
        # print(threshold_kat_cnt)
        normlist= getnorm(listex2)
        if(normlist):
            caseler_norm(kategori_max,normlist)
        else:
            print("hiçbir kategoriye ait değildir")
          
        if say == 5:
           print(df.text[103203 + i], "-> Bu şikayet hiçbir kategoriye ait değil.")  
    else:
        print("bu cümle olumludur")
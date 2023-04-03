# -*- coding: utf-8 -*-
def bert_kategorize(sample):
  import pandas as pd
  df3 = pd.read_csv("bert_cumleler_yeni_91.csv")

  # to remove emojis
  import re
  import pickle
  from emot.emo_unicode import UNICODE_EMOJI,EMOTICONS_EMO

  def remove_emoji(string):
      emoji_pattern = re.compile("["
                             u"\U0001F600-\U0001F64F" # emoticons
                             u"\U0001F300-\U0001F5FF" # symbols & pictographs
                             u"\U0001F680-\U0001F6FF" # transport & map symbols
                             u"\U0001F1E0-\U0001F1FF" # flags (iOS)
                             u"\U00002702-\U000027B0"
                             u"\U000024C2-\U0001F251"
                             "]+", flags=re.UNICODE)
      return emoji_pattern.sub(r'', string)

  def veri_onisleme(veri_text):
      veri_text =remove_emoji(veri_text)
      veri_text =veri_text.lower()
      return veri_text

  for i in range(df3.text.size):
    df3.text[i]=veri_onisleme(df3.text[i])

  from sentence_transformers import SentenceTransformer
  from numpy.linalg import norm
  import numpy as np

  model_bert = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')

  embeddings = model_bert.encode(df3.text)
  # print(embeddings)

  x_train_b= embeddings

  kumas_vector_b = np.empty((30, 768), np.float32)
  renk_vector_b = np.empty((8, 768), np.float32)
  kalip_vector_b = np.empty((24, 768), np.float32)
  gorsel_vector_b = np.empty((16, 768), np.float32)
  kargo_vector_b = np.empty((11, 768), np.float32) #

  for i in range(30):
    kumas_vector_b[i] = x_train_b[i]

  y=0
  for i in range(30,38):
    renk_vector_b[y] = x_train_b[i]
    y=y+1

  y=0
  for i in range(38,62):
    kalip_vector_b[y] = x_train_b[i]
    y=y+1

  y=0   
  for i in range(62,78):
    gorsel_vector_b[y] = x_train_b[i]
    y=y+1

  y=0
  for i in range(78,89):
    kargo_vector_b[y] = x_train_b[i]
    y=y+1


  categories_names = ["Kumas ve dikiş","Renk","Beden ve kalıp","Görselle alakası yok","Kargo ve teslimat"]
  kategoriler_b = []
  kategoriler_b.append(kumas_vector_b)
  kategoriler_b.append(renk_vector_b)
  kategoriler_b.append(kalip_vector_b)
  kategoriler_b.append(gorsel_vector_b)
  kategoriler_b.append(kargo_vector_b)


  def normalizasyon(x,xmax):
    xmin=0
    son= (x-xmin)/(xmax-xmin)
    return son


  def getnorm_bert (liste):
    normlist=[]
    if(liste.count(0)!=5):
      normlist.append(normalizasyon( liste[0],30))
      normlist.append(normalizasyon( liste[1],8))
      normlist.append(normalizasyon( liste[2],24))
      normlist.append(normalizasyon( liste[3],16))
      normlist.append(normalizasyon( liste[4],11))
    print(normlist)
    return normlist

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
        birinci = categories_names[maxlar1[0]]
        
        if((max1-max2<0.3 and max2!=0)):
          if(len(maxlar2)==1):
              ikinci = categories_names[maxlar2[0]]
          else:
            for i in range(len(maxlar2)):
              maxlar2ort.append(np.mean(kategori_list[maxlar2[i]])*0.3 + np.max(kategori_list[maxlar2[i]])*0.7)

            maxlar2ort_new = maxlar2ort.copy()
            maxlar2ort.sort(reverse=True)
            for i in range(len(maxlar2)):
              if(maxlar2ort[0]==maxlar2ort_new[i]):
                ikinci = categories_names[maxlar2[i]]

      else:
        for i in range(len(maxlar1)):
          maxlar1ort.append(np.mean(kategori_list[maxlar1[i]])*0.3 + np.max(kategori_list[maxlar1[i]])*0.7)

        maxlar1ort_new = maxlar1ort.copy()
        maxlar1ort.sort(reverse=True)
        for i in range(len(maxlar1)):
          if(maxlar1ort[0]==maxlar1ort_new[i]):
            birinci = categories_names[maxlar1[i]]
            maxlar1ort[0]=0
        maxlar1ort.sort(reverse=True)
        for i in range(len(maxlar1)):
          if(maxlar1ort[0] == maxlar1ort_new[i]):
            ikinci = categories_names[maxlar1[i]]

    else:
      print("hiçbir kategoriye ait değildir.")
    print("birinci: "+birinci)
    if(len(ikinci)>3):
      print(" ikinci: " +ikinci)


  
  def hesapla(test_vector):
    tres_say=0
    THRESHOLD = 0.5
    liste=[]
    kategori_simlist = []
    for i in range(5):
      tres_say=0
      cosine = np.dot(kategoriler_b[i],test_vector)/(norm(kategoriler_b[i], axis=1)*norm(test_vector))
      kategori_simlist.append(cosine)

      for j in range(len(kategoriler_b[i])):
        if( kategori_simlist[i][j]> THRESHOLD):
          tres_say+=1
      liste.append(tres_say)

    for k in range(5):
      # print("-------------------------------------------------------")
      # print("Kategori: ",categories_names[k])
      print(categories_names[k],"kategorisi için benzerlikler : ",kategori_simlist[k])
    print(liste)
    print("normalizasyonlu sonuçlar:")
    normlist= getnorm_bert(liste)
    if(normlist):
      caseler_norm(kategori_simlist,normlist)
    else:
      print("Hiçbir kategoriye ait değildir.")  

  test = sample
  print("-------------------------------------------------------------")
  print("BERT ile Kategorize Hali")
  print("-------------------------------------------------------------")
  print("test edilen cümle:",test)
  get_test_vector = model_bert.encode(test)
  hesapla(get_test_vector)
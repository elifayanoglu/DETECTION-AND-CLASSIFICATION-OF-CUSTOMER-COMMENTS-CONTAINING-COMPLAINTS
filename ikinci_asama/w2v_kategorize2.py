def ikinci_asama(ornekler):    
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

    # WORD2VEC 
    W2V_SIZE = 300
    W2V_WINDOW = 7
    W2V_EPOCH = 16
    W2V_MIN_COUNT = 1

    # KERAS
    SEQUENCE_LENGTH = 300
    EPOCHS = 3
    BATCH_SIZE = 256


    THRESHOLD = 0.6

    JAVA_HOME = "C:/Users/ASUS/Dropbox/My PC (LAPTOP-CJM5QHNB)/Desktop/jdk-15.0.2_windows-x64_bin.exe"


    # Set log
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # path = "toplam_set_103303.csv"
    path = "normalized_dataset-enyeni.csv"
    test = "test_normal_birlesti.csv"
    df = pd.read_csv(path)

    pd.set_option("display.max_colwidth",None)


    # TEXT CLENAING
    TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^a-zA-Z0-9ğüşöçıİĞÜŞÖÇ]+"

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

    w2v = Word2Vec(documents, vector_size=300, window=W2V_WINDOW, min_count=1, workers=8)

    w2v.build_vocab(documents)

    w2v.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df.text)

    vocab_size = len(tokenizer.word_index) + 1
    print("Total words", vocab_size)

    x_train = pad_sequences(tokenizer.texts_to_sequences(df.text), maxlen=SEQUENCE_LENGTH)


    kumas_dikis = ["ince","incecik","inceydi","yırtık","delik","çekti","yırtıldı","sökük","kalitesi","kaliteli","kalitesiz","gösteriyor","naylon","kalın","yamuk","dandik",
    "dikiş","dikişleri","dikişi","dikmişler","defolu","terleten","terletiyor","terletecek","terletir","kumaş","kumaşı","kumaşını","kumaşın","küçüldü","kayboldum"]


    kumas_vector = []
    for i in range(len(kumas_dikis)):
        string = kumas_dikis[i]
        # print(string)
        kumas_vector.append(w2v.wv[string])
        # print(kumas_vector[i])

    renk = ["rengi","rengini","renginin","soluk","solmuş","soluyor","soldu","canlı"]


    renk_vector = []
    for i in range(len(renk)):
        string = renk[i]
        # print(string)
        renk_vector.append(w2v.wv[string])
        # print(renk_vector[i])

    kalip_beden = ["bol","boldu","büyük","küçük","dardı","dar","geniş","pot","potluk","kesiminde","kesiminden","oversize","kesim","kesimi","kesimleri","kolları","uzun","kısa",
    "boyu","kalıbı","kalıp","kalıbını","beden","bedeni","bedenim","bedene","bedenler","bedenleri"]


    kalip_vector = []
    for i in range(len(kalip_beden)):
        string = kalip_beden[i]
        # print(string)
        kalip_vector.append(w2v.wv[string])
        # print(kalip_vector[i])

    gorselle_alaka = ["alakası","fotoğraf","fotoğrafta","fotoğraftaki","fotoğraftakinden","fotoğrafla","fotoğraftakiyle","göründüğü",
    "görseldeki","görseldekiyle","görselle","görsel","resimdeki","resimdekinden","resimdekiyle","resimde"]

    gorsel_vector = []
    for i in range(len(gorselle_alaka)):
        string = gorselle_alaka[i]
        # print(string)
        gorsel_vector.append(w2v.wv[string])
        # print(gorsel_vector[i])

    kargo = ["teslimat","yavaş","geç","paketleme","kargo","leke","lekeli","etiketsiz","kusurlu","yanlış","eksik","yerine"]

    kargo_vector = []
    for i in range(len(kargo)):
        string = kargo[i]
        # print(string)
        kargo_vector.append(w2v.wv[string])
        # print(kargo_vector[i])


    from numpy.linalg import norm
    categories_names = ["kumas_vector","renk_vector","kalip_vector","gorsel_vector","kargo_vector"]
    kategoriler = []
    kategoriler.append(kumas_vector)
    kategoriler.append(renk_vector)
    kategoriler.append(kalip_vector)
    kategoriler.append(gorsel_vector)
    kategoriler.append(kargo_vector)

    kategori_isimleri = ["kumas ve dikiş","renk","beden ve kalıp","görselle alakası yok","kargo ve teslimat"]


    def getnorm_w2v (liste):#NORMALİZASYON İÇİN 
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

    
    for i in range(len(ornekler)):

        #örnek cümle ile test
        # sample = "almanizi tavsiyye etmiyorum çünkü çok kalitesiz duruyor ve rengi göründüğü gibi değil geldiğinde kötü bir koku vardı ama çok ince geldi iade edeceğim" 
        sample = ornekler[i]
        sample2=sample

        # sample = testler[i]

        import normalization_pb2 as z_normalize
        import normalization_pb2_grpc as z_normalize_g
        import grpc
        import pandas as pd

        channel = grpc.insecure_channel('localhost:6789')

        norm_stub = z_normalize_g.NormalizationServiceStub(channel)

        def normalize(i):
            response = norm_stub.Normalize(z_normalize.NormalizationRequest(input=i))
            return response.normalized_input
  
        sample = normalize(sample)
        sample2 = normalize(sample2)
        print("-------------------------------------------------------")
        print("normal sentence:", sample)

        sentence = preprocess(sample)

        tokenize = sentence.split()


        tokenize_vector = [] 
        for i in range(len(tokenize)):
            string = tokenize[i]
            tokenize_vector.append(w2v.wv[string])
            # print(tokenize_vector[i])
        
        print("-----------------------------------------------------")
        print("W2V ile Kategorize Hali")
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
        new_list = []
        new_list = threshold_kat_cnt.copy()

        for a in range(5):
            print(a+1,". kategori için benzerlikler: ",kategori_max[a])
        print("test edilen cümle: ",sample)
        import copy
        listex=copy.deepcopy(threshold_kat_cnt)
        listex2=copy.deepcopy(threshold_kat_cnt)
        print("normalizasyonlu sonuçlar:")
        normlist= getnorm_w2v(listex2)
        if(normlist):
            caseler_norm(kategori_max,normlist)
        else:
            print("hiçbir kategoriye ait değildir")   
       

# ornekler = ["Fiyatı uygun göründüğü gibi kaliteli değilll",
#             "Hiç göründüğü gibi değil. Bedeni çok büyük. En az 2 beden. Dokusu rengi ve duruşu kötü. İade ettim.",
#             "Malesef beklediğim kadar kaliteli gelmedi bana kumaşı . Çok fazla da salaş . İade etmeye uğraşamayacağım için belki giyerim düşüncesiyle iade etmeyeceğim ama bu paraya çok daha güzel parçalar alınabilir . Pandemi dolayısıyla da ürün elime neredeyse 2 hafta sonra ulaştı",
#             "Tek kullanımlık yıkandıktan sonra giyilmez",
#             "Beden yanlış gönderildi, ürün elime geri geldi, kalitesi de olabildiğince kötü aylardır ürün elimde ve hala iade yapamadım. 1 sene sonra olur herhalde...",
#             "Urunun düğme kısımlarında mavi tukenmez kalemle çizilmiş çizgiler vardi. Ayrica epey salaş oldu üzerime. Bu sebeple iade ettim",
#             "Ürün güzel ama xs almama rağmen aşırı büyük durdu asla giyemeyeceğim bir şey iadem kabul olursa mutlu olcam",
#             "Aşırı boooool M istememe rağmen L gibi oldu",
#             "Ürünü beğenmedim üzerinde çizikler vardı ve xs almama rağmen aşırı büyük geldi iade ettim",
#             "Ürün m beden aldım çok büyük ve hep kalem lekeli geldi Aradım tekrar gönderin dediler ve yardımcı oldular",
#             "Gerçekten pes beden olarak XL seçmeme rağmen biri M gelmiş biri L bravoooooo İadede edemiyorum şimdi ne yapıcam ben bunları? tebrikler trendyol",
#             "kalıbını beğenmeyip iade ettim.",
#             "Ürün genel anlamda duruşu kumaşı güzel fakat lekeli geldi keşke kontrol etseydiniz.",
#             "Bu lekeler nedir ya. Hiç mi kontrol edilmiyor. Xs aldığım halde omuzları oturmadı. Beden çok bol.",
#             "beğenmedim kumaşni filan"]

ornekler = ["Malesef iade ettim ürünü M beden söyledim kocaman geldi ve kumaşı cok inceydi bu fiyatta etmezdi bence",
"icini gosteriyor l beden giyene m beden buyuk geliyor iade ettim",
"38 bedenim ama 34 beden söyledim aşşiiiriii büyük bi gömlek ve çok ince. parası için değmez",
"Kumaşı okul gömleği gibi kalitesiz, cebi, kolları duruşu biraz ucuz duruyor. İade ediyorum.",
"Kargo çok hızlı geldi. Ama ürünün kalitesi kötü, bana göre. Çok ince. Fotoğraftaki gibi değil. 170 boyum 52 kg, beden M aldım - tam oversize gibiydi. Ama iade edildim.",
"Duruşu ve bedenleri kötü. Beklediğim gibi bir gömlek gelmedi iade ettim",
"Kumaşı pamuklu yapısı ince ve iç belli ediyor 36 beden almıştım fazla oversize geldi iade yapacağım",
"Malesef kendi bedenimi almama rağmen hic bol görünüm alamadım, kalitesi idare eder açıkçası lise gömlegi görünümünde oldu. İade edeceğim.",
"Boyu ve genişliği orantılı değil onun dışında güzel bir ürün iç de gösteriyor",
"çok bol arkasi pot durdu kumaşı çok ince",
"Kumaş kalitesi ortalamanın altında ve kalıbı bol",
"Hiç göründüğü gibi değil, kısa ve küçük kumaş ince",
"Kumaşın kalitesi ve modeli güzel değildi iade ettim üzülerek",
"çok büyük bir ürün üstünüzde çuval gibi duruyor kumaşı kalitesiz",
"Dikişleri ve kumaşı çok kalitesiz kalıbı da kötü",
"Kumaşını beğenmedim. Boyfriend değil bence önü kısa arkası uzun. Hiç güzel bir gömlek değil tavsiye etmiyorum. Bedeni de geniş değil",
"kumaşı çok sert dümdüz bir beyaz gömlek tavsiye etmem kalıbı kötü",
"Kalıbı çok geniş.70 kiloya 40 beden aldım.Çok büyük geldi.İade ettim Kuması çok ınce",
"Duruş olarak berbat ve ötesi. Kumaşı da beğenmedim",
"Gömleğin arkadan uzunluğu çok dar düşüyor duruşu çok güzel değil. lekeli geldi yıkadım ama çıkmadı",
"Çok kalitesiz ve en küçük bedeni almama rağmen aşırı bol geldi",
"Çok ince bir kuması var. Biraz da ucuz duruyor. Asla oversize değil, kesinlikle büyük alın. İade ettim.",
"Kumaşından hoşlanmadım çok naylonumsu ve kargo yavaştı",
"Berbat bir gömlek.Kalıbı kötü önü çok uzun arkası çok uzun sünnet çocuğu gömleği gibi,okul gömleği gibi kumaşı da parlak",
"Battal boy, kalitesiz bir ürün. Kimselere tavsiye etmem.",
"gömlek naylon gibi baya da büyük ona göre alın.",
"Kumaş kalitesi iyi değil ve Aşırı bol durdu kendi bedenimi istememe rağmen",
"fotograftakiyle alakasi yok, normalde 36-38 giyerim bol dursun diye 40 aldim kisacik durdu,kollari kisa",
"Tam bir hayal kırıklığı oldu. 34 bedeni bile aşırı bol ve kesimi bir şeklisiz. Sırt kısmında toplanma yapıyor. Kumaşı ucuz okul gömleği kumaşından. Genel olarak gömlek ucuz duruyor.",
"1.60 biy 57 kiloyum..s aldim cok buyuk sx alinmaliymis ayrica kumaşı kötü aşırı çabuk kırışıyo hışır hışır iade 👎🏼",
"Boyu ve duruşu aşırı kötü bu kadar kötü bir gömlek görmedim. Nesini beğenmişler anlamadım aşırı inçe kumaş çok dandik",
"ben pek beğenemedim kumaşı çok ince ve mankendeki gibi durmuyor",
"ic gosteriyor ve cok uzun.",
"Ürün hiç göründüğü gbi değil herzamanki bedenimi aldım 36 boyfriend bol ve geniş olması gerekirdi ama hiç öyle değildi ",
"S BEDEN ALDIM AŞIRI BOL VE KUMAŞI ÇOK KALİTEİSİZ",
"Kolları kısa ve kumaşı kalitesiz. İade",
"Kumaşını da duruşunu da hiç beğenmedim",
"34 beden aldım Tunik gibi uzun kolları kısa. Değmez kalite sıfır",
"Yeri çok ince ve bildiğiniz okul gömleğinden farksız çok kalitesi duruyor bilginize iade ettim direk.",
"Kargo geç geldi. Okul gömleği gibi çok sert ve kalitesiz kumaşı bu yüzden iade ettim",
"çok kötü iç gösteriyo ve kısa",
"İç gösteriyor, beğenmedim. Yıkanınca rengi soldu",
"Malı çok kötü iade ettim. Kargo yavaş",
"Kalitesi çok kötü sanki elek gibi 😔 Üzerinde bir kaç yerinde siyah lekeler vardı.",
"34 beden almama rağmen gerçekten üzerimde çuval gibi duran bir ürün oldu. kargo geç geldi. Maalesef iade edeceğim.",
"İlk yıkamada hemen çekti bu ne ya. duruşu da kötü",
"Naylon gibi ürün. kesimi de kötü",
"fena değil iade etmedim indirimden aldığım için ama boyum uzun olduğu için kolları aşırı kısa kaldı bedeni büyük söylesem çok salaş duracaktı. Çok kırışan bir kumaşı var",
"Ürün fotoğraftaki gibi değil ben daha salaş beklemiştim fakat lise gömleği gibi geldi. 1 beden büyük söylememe rağmen istediğim oversize görüntüyü alamadım.",
"Bollll ve kumaş biraz kalitesiz 😔😔",
"Kumaşı pek kaliteli değil. Duruşu idare eder. Alınmasa da olur.",
"Beden yanlış gönderildi, ürün elime geri geldi, kalitesi de olabildiğince kötü aylardır ürün elimde ve hala iade yapamadım. 1 sene sonra olur herhalde...",
"Lekeli ve defolu geldi ayrıca herkes Bi beden küçük alın demiş öyle yaptım bu sefer de küçük geldi kolları boyu falan boyum 172 kilom 55 iade edicem",
"Ceket kolları bedene göre aşırı kısa hiç beğenmedim ve çok kalın hiç tavsiye etmem",
"Ürün kalitesi çok kötü. Bildiğiniz naylon kumaş. Ve paketleme alırı özensiz ve kırış kırış geldi.",
"Rezalet. Ürün asla mankendeki gibi değil kumaşı çok kalitesiz boyunu neye göre yapmışlar anlayamadım aşırı uzun asla tavsiye etmem aşırı kötü bir ürün",
"begenmedim, naylonsu bir kumaşı var. rengi cok kötü kolay kolay kombin olmayacak bir gri. cok acık renk. kaitesiz duruyor.",
"Uzgunum ama defolu ürün göndermişler. Cep kısmı fotoğraftaki gibi değildi koltuk altı kısmı sokulmustu ben iade edeceğim ama kesinn başkası sipariş verince ona gönderecekler alırken 2 kez düşünün",
"Pantolonla takım almama rağmen ton farkı var ve ceket çok büyük 165 boy 58 kilo 38 beden çuval gibi oldu",
"Kesinlikle bir beden küçük alınmalı belim kalın olduğu halde büyük geldi, boyum 168 paçaları çok uzun, rengi kötü bir kahverengi asla fotodaki gibi değil, olduğumdan daha kilolu gösterdi o yüzden iade",
"160 68 kilo 40 beden aldım dar geldi belden kapanmadı.  Kumaşı biraz sert ve paçalarıda uzundu iade .",
"Kalıbı çok dar bildiğin polyester",
"Ürün resimde görüldüğü gibi güzel durmuyor giyildiğinde kumaşı kötü değil giyilebilir fakat boyu resimdeki gibi kısa değil uzun kestirmek istiyor kendi bedenimi aldım bi beden büyük geldi o yüzden bir beden küçüğü alınabilirmiş duruşunu beğenmediğim için iade ettim",
"yırtık ve lekeli",
"kocaman bi leke vardi iade ettim ve kalıpları cidden cok buyuk",
"Ürün elime ulaştığında ortasında sararmış bi leke vardı. Paketlerken görünmeyecek gibi değil. Nasıl dikkat etmezler anlamadım ayrıca kumaşıda bi yıkamada yıpranacak bi kumaş dont!",
"Çok buruşan bir kumaşı var, aşırı kalitesiz. Kesinlikle almayın. Kargo da çok yavaş.",
"3 renk birden aldım sadece bu siyah olan görseldeki gibiydi ama kendi bedenimi almama rağmen çok boldu ürünü iade ettim",
"görseldekiyle hiç alakası yok kumaşı çok kötü ayrıca iç gösteriyor S beden almama rağmen büyük geldi iyade etim",
"Ürün elime kusurlu ulaştı. Göğüs kısmında siyah lekeler var ve dikişi çok çok kötü. Fotoğraflardan detaylı görebilirsiniz. İade 👎🏻",
"Beden olarak L söylememe rağmen sanki bir çocuk için dikilmiş de giymişim gibi oldu. Askı boylarından biri kısa biri uzun, etek boyu kesinlikle fotoğraftakinden kısa, göğüs kısmı B kup giymeme rağmen aşırı bastı ve çok yukarıda kaldı. Kumaşı kötü. Kesinlikle değmez.",
"Resimdeki leopar ile gercekte urundeki leopar renkleri oldukca farkli resimde kahve tonlarinda iken grlen daha soluk kahve sari tonlarinda",
"42 aldım büyük geldi iade ettim 40 sipariş verdim tam oldu ama renk aynı renk deil 42 deki resimdeki gibiydi gelen 40 beden rengi solmuş hali geldi bu nedir acaba satıcıya soruyorum😲",
"Hayatımda bu kadar kötü bir milla ürünü görmemiştim. Rengi solmuş kumaşı tamamen naylon dikişleri aşırı derecede kötü iade",
"Ürün malesef hatalı geldi .Dikişleri çok kötü.Kol oyuntusu bi tarafın farklı ve pot duruyor ...",
"Kumaşın görsel ile ve kumaş bilgisi ile alakası yok. Kumaş viskon yazıyor ama polyesterli dükümsüz bambaşka bir kumaş. Geç teslimat ve farklı kalite ürün, yazık😔",
"Kumaşlarını beğendim fakat beyaz olanın yakasında leke var ve siyah olan etiketsiz geldi. Hızlı teslimat dolayısıyla ürünler kontrol edilmeden mi gönderiliyor? Hayal kırıklığı..",
"ürün hem sökük hem etiketsiz geldi iade edeceğim maalesef. ama modeli duruşu güzel yalnız kumaşı ince bi tık kalitesiz",
"Ürünü rengi güzel bedene göre duruyor ama arkada çok pot duruyor.",
"Rengi kırmızı görünüp aldığım da pembe olarak gelmesi çok kötü beğenmedim alacak arkadaşlara tavsiyem fotoğraftaki ile gerçekteki çok farklı",
"Ürünün rengi ve kumaşı göründüğü gibi değil. Hiç sevemedim.",
"Dikişleri o kadar uyduruk ki bana gelmeden sökülmüşte gelmiş. Keske ürünleri kargolamadan kontrol etseniz bi genel olarak o ürün o sökükle müşteriye gönderilmez. İADE....",
"dikiş berbat renk cok kotu hic göründüğü gibi değil son derece kalitesiz",
"Ürünü tavsiye ve yorumlara dayanarak almıştım. Fakat ürünü dikiş hatası ve sol arka kol kesiminde bulunan bi leke nedeni ile iade ediyorum.",
"Ürünün malı pek kaliteli değil, 168 boyundayım 58 kiloyum S bedenini aldım kocaman oldu Xs bedenini almanızı tavsiye ederim, ürünün kesiminde yamukluk olduğu ve çok büyük olduğu için malıda güzel olmadığı için iade ettim",
"Fotolarda daha güzel duruyordu ama kumaşı terletecek cinsten ayrıca eteğinde yırtıkla geldi iade etmekle uğraşmamak için evde giyeceğim",
"kalitesiz, göğüs dar kollar uzun ve bi tarafı yırtık geldi berbat. karantinada olduğumuz için iade edemedim ama kullanacağımı sanmıyorum.",
"arkadaşlar ürün defolu geliyor dikişlerin yarısı ters yarısı düz dikilmiş yani etek kısmında iç tarafta olması gereken dikiş ön yüzde..ters çevirip giyeyim dedim bu sefer kolları ters dikiş oluyor..berbat bir ürün.. kazancınız helal değil.tabi kimin umrunda",
"elbisenin gorseldeki ile alakasi yok çok kötü kumaşı çok sıradan penye kumas resimdeki elbise çok daha şık görünüyor",
"Görsel ile ilgilisi olmayan bir elbise geldi içinde astarı yok et uçlarıda görseldeki ile aynı değil ne yazık ki",
"kumaşı çok İnce kargo çok geç geldi",
"Göründüğü gibi simsiyah değil daha soluk duruyor. İade ettim.",
"Kumaşı çok adi ,rengi bi yıkamada gitti.Temizlik bezi diye kullanmak istiyorsanız alabilirsiniz.",
"O kadar ince bir kumaşı var ki içiniz görünüyor resmen,bedeni olması gerektiğinden daha küçük o yüzden iade ettim.",
"Ürünüm eksik geldi kazağamın altını dokumamışlar. Mankendeki göbeğinin altında s beden model, sipariş ettiğim l beden 60 kiloyum riske girmeyeyim dedim fakat gelen göbek üstü s beden gibi... kesinlikle fotoğraftaki gibi değil.",
"Ürünün defolu olduğunu yıkandıktan sonra farkettim malesef yoksa direk iade ederdim. Kazağın arkasının tam orta kısma gelen yerinde kırmızı iple dikilmiş bir kısım var. Asla almayın tavsiye etmiyorum. Mavi rengini de almıştım orantısız diye iade ettim ama bu kadar da kalitesizlik olmaz.",
"Berbat boyu görünenden çok daha kısa ve düzgün dikilmediği için iç gösteriyor. aldigim gibi iade ettim .Rezalet icinin dikisleri sökülmüs bir sekilde geldi🤢🤮",
"Çok kalitesiz yorumlara ve fotoğraflara aldanıp aldım ama hiç güzel değil direk iade",
"İğrenç bir ürün. Dikişleri iğrenç. Geldiğinde önünde bir sürü leke vardı. Bu kadar kötü duran bir kazak görmedim. Çok ama çok kötü. Direk iade edeceğim. Kalitesiz duruyor, hem de çok!",
"ürün çok baştan savma yapılmış. bel lastiği çok kalın ve aşırı sıkıyor. kumaşı da çok uzun dizlerden aşağıda kalıyor. ve gereksiz şekilde bol yapılmış.",
"gelen ürünle fotoğraftaki ürünün rengi uyuşmuyor yosun yeşili mi desem değişik kötü bir renk geldi bence kalın yorumlarda yazlık yazmışlar yaz için fazla kalın iade edicem"]

ikinci_asama(ornekler)
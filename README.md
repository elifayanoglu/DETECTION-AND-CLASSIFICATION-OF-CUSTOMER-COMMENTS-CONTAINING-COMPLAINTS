# complaint-classification
## EN:
This project consists of two phases. The first stage is to determine whether customer comments contain complaints or not by sentiment analysis.
The second step is to classify the comments found to contain complaints by measuring their similarities to 5 predetermined categories.

### Usage
Inside the first_asama folder, sentences are split into tokens using Word2Vec, BERT and BOW methods.
Then, using SVM, LR, CNN + LSTM and MNB classifiers for these methods, respectively, the method that gave the most successful results was used in the second stage.

In the second_stage folder, we save the most successful model we obtained in the first stage with dump.py.
There are functions required for loading and categorizing the "weights.h5" weight that we have registered with cnn_pickle.py.
All the steps we have implemented for W2V with w2v_kategorize2.py have been combined into a single function and tested with examples.
A version tested with BERT is available in the cosine_sim_emb91.py file.




# şikayet-sınıflandırma
## TR
Bu proje iki aşamadan oluşmaktadır. Birinci aşama, duygu analizi ile müşteri yorumlarının şikayet içerip içermediğinin tespitini yapmaktadır.
İkinci aşama ise şikayet içerdiği tespit edilen yorumların önceden belirlenmiş olan 5 adet kategoriye olan benzerlikleri ölçülerek sınıflandırılmasıdır.

### Kullanım
birinci_asama klasörünün içinde Word2Vec, BERT ve BOW yöntemleri kullanılarak cümleler tokenlarına ayrılmıştır. 
Daha sonra sırasıyla bu yöntemler için SVM, LR, CNN + LSTM ve MNB sınıflandırıcıları kullanılarak en başarılı sonucu veren yöntem ikinci aşamada kullanılmıştır.

ikinci_asama klasörünün içinde dump.py ile birinci aşamada elde ettiğimiz en başarılı modeli kaydediyoruz.
cnn_pickle.py ile kaydetmiş olduğumuz "weights.h5" ağırlığının yüklenmesi ve kategorileme için gerekli fonksiyonlar yer almaktadır.
w2v_kategorize2.py ile W2V için uygulamış olduğumuz bütün aşamalar tek bir fonksiyon haline getirilip örneklerle test edilmiştir.
cosine_sim_emb91.py dosyasında BERT ile denenmiş versiyonu mevcuttur.


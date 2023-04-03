import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk

nltk.download('stopwords')
data_path = "toplam_set_104937.csv"
punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

stopWords = set(stopwords.words('turkish'))


def veri_onisleme(veri_text):
    # print(veri_text)
    for ele in veri_text:
        if ele in punc:
            veri_text = veri_text.replace(ele, "")

    split_words = []
    # print(veri_text)
    for w in veri_text.split():
        if w.lower() not in stopWords:
            split_words.append(w)

    print(split_words)
    return split_words


def main():
    # Load dataset
    datafile = pd.read_csv(data_path)

    datafile.drop_duplicates(inplace=True)

    messages_bow = CountVectorizer(analyzer=veri_onisleme).fit_transform(datafile['text'])

    print(messages_bow)

    text_train, text_test, label_train, label_test = train_test_split(messages_bow, datafile['pos'], test_size=0.20,
                                                                      random_state=0)

    classifier = MultinomialNB().fit(text_train, label_train)

    # pred = classifier.predict(text_train)
    # print(classification_report(label_train, pred))
    # print('\nConfic: matrix:\n', confusion_matrix(label_train,pred))
    # print('accuracy:', accuracy_score(label_train,pred))

    pred = classifier.predict(text_test)
    print(classification_report(label_test, pred))
    print('\nConfic: matrix:\n', confusion_matrix(label_test, pred))
    print('accuracy:', accuracy_score(label_test, pred))


if __name__ == "__main__":
    main()

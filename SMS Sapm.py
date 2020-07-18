import nltk
import random
import pickle
import numpy as np
from nltk.corpus import stopwords
import spacy
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import time

with open('SMSSpam.txt', 'r') as SMS:
    documents = SMS.read().split('\n')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

documents = np.array(documents)
documents = np.char.split(documents, sep='\t')

"""USE SKLEARN MODEL ONLY"""
doc = documents
random.shuffle(doc)
x_train = np.array([i[1] for i in doc][:4500])
y_train = np.array([i[0] for i in doc][:4500])
x_test = np.array([i[1] for i in doc][4500:])
y_test = np.array([i[0] for i in doc][4500:])

ham, spam = [], []
for i in range(len(x_train)):
    if y_train[i] == 'ham':
        ham.append(x_train[i])
    else:
        spam.append(x_train[i])
stopwords1 = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords1).generate(' '.join(ham))
wordcloud2 = WordCloud(stopwords=stopwords1).generate(' '.join(spam))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.imshow(wordcloud2, interpolation='bilinear')
plt.axis("off")
plt.show()
# print(wordcloud.words_)


def stem(sent):
    stemmed_sentence = []
    for w in sent:
        word = nltk.word_tokenize(w.lower())
        word = [wo for wo in word if wo.isalnum()]
        filtered_sentence = [j for j in word if j not in stop_words]
        stemmed_words = ([ps.stem(k) for k in filtered_sentence])
        stemmed_sentence.append(' '.join(stemmed_words))
    return stemmed_sentence

start_time = time.time()
tfidfconverter = TfidfVectorizer(ngram_range=(1, 1), max_features=12000, max_df=0.4, stop_words=stop_words)
traindataset = tfidfconverter.fit_transform(stem(x_train))
randomclassifier = RandomForestClassifier(n_estimators=4000, criterion='entropy')

randomclassifier.fit(traindataset, y_train)
testdataset = tfidfconverter.transform(stem(x_test))
predictions = randomclassifier.predict(testdataset)

accuarcy = accuracy_score(y_test, predictions)
classification = classification_report(y_test, predictions)
matrix = confusion_matrix(y_test, predictions)
print(accuarcy)
print(classification)
print(matrix)
print("%s seconds" % (time.time() - start_time))

# labels = []
# sms = []
# for i in range(len(documents)):
#     labels.append((documents[i][0]))
#     sms.append((documents[i][1]))
# nlp = spacy.load('en_core_web_sm')
# df = pd.DataFrame(labels, columns=['Labels'])
# df['SMS'] = sms
# for i in nlp.pipe(df['SMS']):
#     for word in i:
#         print(word.text,  word.pos_, word.dep_)

"""CREATE BAG OF WORDS MANUALLY AND NLTK CLASSIFIER WITH SKLEARN MODELS"""
random.shuffle(documents)
all_words = []
for w in documents:
    word = nltk.word_tokenize(w[1])
    word = [wo.lower() for wo in word if wo.isalnum()]
    filtered_sentence = [j for j in word if j not in stop_words]
    stemmed_sentence = [ps.stem(k) for k in filtered_sentence]
    for p in stemmed_sentence:
        all_words.append(p)

all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = nltk.word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

featuresets = [(find_features(rev), category) for (category, rev) in documents]
random.shuffle(featuresets)

training_set = featuresets[:4500]
testing_set = featuresets[4500:]

# classifier_f = open("Wilfred_naivebayes.pickle", "rb")
# classifier = pickle.load(classifier_f)
# classifier_f.close()
#
# classifier = nltk.NaiveBayesClassifier.train(training_set)
# print("Classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
# classifier.show_most_informative_features(15)

start_time = time.time()
classifier = nltk.SklearnClassifier(RandomForestClassifier(n_estimators=4000, criterion='entropy'))
classifier.train(training_set)
print("Classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
print("%s seconds" % (time.time() - start_time))
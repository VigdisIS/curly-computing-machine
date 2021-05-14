import time

import nltk
import pandas as pd
import re
#import preprocessor as p
from nltk.classify import svm
from nltk.corpus import stopwords

nltk.download('stopwords')
from nltk.stem import PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report

tweets = pd.read_csv('./datasets/sentiment_dataset.csv')
#x = tweets['text']

'''
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(tweets['cleaned_text'], tweets['target'], test_size=0.3)

from sklearn.feature_extraction.text import TfidfVectorizer
# Create feature vectors
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
train_vectors = vectorizer.fit_transform(X_train.values.astype('U'))
test_vectors = vectorizer.transform(X_test.values.astype('U'))


import time
from sklearn import svm
from sklearn.metrics import classification_report
# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(train_vectors, y_train)
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1
# results
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
report = classification_report(y_test, prediction_linear, output_dict=True)
print('positive: ', report['4'])
print('negative: ', report['0'])


'''




vectorizer = CountVectorizer()

# tokenize and make the document into a matrix
X_fin = vectorizer.fit_transform(tweets['cleaned_text'].values.astype('U'))


from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_fin, tweets['target'], test_size=0.3)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
from sklearn.metrics import classification_report

cf = classification_report(y_test, y_pred)
print(cf)


import pickle
# pickling the vectorizer
pickle.dump(vectorizer, open('sentiment_vectorizer.sav', 'wb'))
# pickling the model
pickle.dump(model, open('sentiment_classifier.sav', 'wb'))
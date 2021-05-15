import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# ML Libraries
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Global Parameters
stop_words = set(stopwords.words('english'))
# tweets = pd.read_csv('../datasets/sentiment_dataset.csv')
# x = tweets['text']


'''
tf=TfidfVectorizer()
text_tf= tf.fit_transform(tweets['cleaned_text'].values.astype('U'))

from sklearn.model_selection import train_test_split

# X -> features, y -> label
X = text_tf
y = tweets['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)



dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
preddt = dt.predict(X_test)
print("Confusion Matrix for Decision Tree:")
print(confusion_matrix(y_test,preddt))

score = round(accuracy_score(y_test,preddt)*100,2)
print("Score:",score)

print("Classification Report:")
print(classification_report(y_test,preddt))



'''
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

'''
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(ngram_range=(1, 2))

vectorized_data = count_vectorizer.fit_transform(tweets['cleaned_text'].values.astype('U'))
#indexed_data = hstack((np.array(range(0, vectorized_data.shape[0]))[:, None], vectorized_data))


def sentiment2target(sentiment):
    return {
        0: 0,
        4: 1
    }[sentiment]


targets = tweets['target'].apply(sentiment2target)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(vectorized_data, targets, test_size=0.3, random_state=0)

from sklearn import svm

clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
'''

'''
from sklearn.model_selection import train_test_split
data_train, data_test, targets_train, targets_test = train_test_split(indexed_data, targets, test_size=0.4, random_state=0)
data_train_index = data_train[:,0]
data_train = data_train[:,1:]
data_test_index = data_test[:,0]
data_test = data_test[:,1:]


from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
clf = OneVsRestClassifier(svm.SVC(gamma=0.01, C=100., probability=True, class_weight='balanced', kernel='linear'))
clf_output = clf.fit(data_train, targets_train)

cf = classification_report(y_test, y_pred)
print(cf)

print(clf.score(data_test, targets_test))

'''


def preprocess_tweet_text(tweet):
    tweet = tweet.lower()

    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))

    # Remove twitter words
    stop_words.add('rt')
    stop_words.add('im')
    stop_words.add('u')

    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]

    #ps = PorterStemmer()
    #stemmed_words = [ps.stem(w) for w in filtered_words]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in filtered_words]

    return " ".join(lemma_words)


def load_dataset(filename, cols):
    dataset = pd.read_csv(filename, encoding='latin-1')
    dataset.columns = cols
    return dataset


def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector

''' 
tweets = pd.read_csv('../datasets/sentiment_dataset.csv')

x = tweets['text']


def clean_tweets(df):
    tempArr = []
    for tweet in df.values:
        processed = preprocess_tweet_text(tweet)
        tempArr.append(processed)
    return tempArr

preprocessed_text = clean_tweets(x)
'''

# Load dataset
dataset = load_dataset('../datasets/sentiment_dataset.csv', ['target', 'user', 'text', 'cleaned_text'])
#Preprocess data
dataset.text = dataset['text'].apply(preprocess_tweet_text)


# Split dataset into Train, Test

# Same tf vector will be used for Testing sentiments on unseen trending data
tf_vector = get_feature_vector(np.array(dataset.iloc[:, 2]).ravel())
X = tf_vector.transform(np.array(dataset.iloc[:, 2]).ravel())
y = np.array(dataset.iloc[:, 0]).ravel()

'''
# Same tf vector will be used for Testing sentiments on unseen trending data
tf_vector = get_feature_vector(np.array(preprocessed_text).ravel())
X = tf_vector.transform(np.array(preprocessed_text).ravel())
y = np.array(tweets['target']).ravel()
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

# Training Naive Bayes model
NB_model = MultinomialNB()
NB_model.fit(X_train, y_train)
y_predict_nb = NB_model.predict(X_test)
print(accuracy_score(y_test, y_predict_nb))

# Training Logistics Regression model
LR_model = LogisticRegression(solver='lbfgs')
LR_model.fit(X_train, y_train)
y_predict_lr = LR_model.predict(X_test)
print(accuracy_score(y_test, y_predict_lr))

import pickle
# pickling the vectorizer
pickle.dump(tf_vector, open('sent_vectorizer.sav', 'wb'))
# pickling the model
pickle.dump(NB_model, open('sent_classifier.sav', 'wb'))









'''


vectorizer = CountVectorizer()

# tokenize and make the document into a matrix
X_fin = vectorizer.fit_transform(tweets['cleaned_text'].values.astype('U'))


from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_fin, tweets['target'], test_size=0.3)


model = BernoulliNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
from sklearn.metrics import classification_report

cf = classification_report(y_test, y_pred)
print(cf)



'''

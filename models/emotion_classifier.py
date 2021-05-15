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


def preprocess_tweet_text(tweet):
    tweet.lower()
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]

    # ps = PorterStemmer()
    # stemmed_words = [ps.stem(w) for w in filtered_words]
    # lemmatizer = WordNetLemmatizer()
    # lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]

    return " ".join(filtered_words)


def load_dataset(filename, cols):
    dataset = pd.read_csv(filename, encoding='latin-1')
    dataset.columns = cols
    return dataset


def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector

tweets = pd.read_csv('../datasets/emotion_dataset.csv')

x = tweets['Tweets']


def clean_tweets(df):
    tempArr = []
    for tweet in df:
        processed = preprocess_tweet_text(tweet)
        tempArr.append(processed)
    return tempArr

preprocessed_text = clean_tweets(x)


# Split dataset into Train, Test

# Same tf vector will be used for Testing sentiments on unseen trending data
tf_vector = get_feature_vector(np.array(preprocessed_text).ravel())
X = tf_vector.transform(np.array(preprocessed_text).ravel())
y = np.array(tweets['Feeling']).ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

# Training Naive Bayes model
NB_model = MultinomialNB()
NB_model.fit(X_train, y_train)
y_predict_nb = NB_model.predict(X_test)
print(accuracy_score(y_test, y_predict_nb))

# Training Logistics Regression model
LR_model = LogisticRegression(solver='lbfgs', max_iter=1000)
LR_model.fit(X_train, y_train)
y_predict_lr = LR_model.predict(X_test)
print(accuracy_score(y_test, y_predict_lr))

import pickle
# pickling the vectorizer
pickle.dump(tf_vector, open('emot_vectorizer.sav', 'wb'))
# pickling the model
pickle.dump(NB_model, open('emot_classifier.sav', 'wb'))




''' 
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(ngram_range=(1, 2))

vectorized_data = count_vectorizer.fit_transform(tweets['cleaned_text'].values.astype('U'))
#indexed_data = hstack((np.array(range(0, vectorized_data.shape[0]))[:, None], vectorized_data))

def sentiment2target(sentiment):
    return {
        'happy': 0,
        'sad': 1,
        'surprise': 2,
        'fear': 3,
        'disgust': 4,
        'angry': 5
    }[sentiment]


targets = tweets['Feeling'].apply(sentiment2target)

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
cleaned_tweets = clean_tweets(x)

tweets['cleaned_text'] = cleaned_tweets

tweets.to_csv('./datasets/emotion_dataset.csv', index=False)
'''


'''



tf=TfidfVectorizer()
text_tf= tf.fit_transform(tweets['cleaned_text'])

from sklearn.model_selection import train_test_split

# X -> features, y -> label
X = text_tf
y = tweets['Feeling']

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



import pickle
# pickling the vectorizer
pickle.dump(tf, open('emot_vectorizer.sav', 'wb'))
# pickling the model
pickle.dump(dt, open('emot_classifier.sav', 'wb'))

'''


'''
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# loading the iris dataset
#iris = datasets.load_iris()

# X -> features, y -> label
X = tweets['cleaned_text']
y = tweets['Feeling']

vectorizer = CountVectorizer()

# tokenize and make the document into a matrix
X_fin = vectorizer.fit_transform(X.values.astype('U'))

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X_fin, y, test_size=0.4)


model = MultinomialNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
from sklearn.metrics import classification_report

cf = classification_report(y_test, y_pred)
print(cf)
'''

'''
# training a DescisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

dtree_model = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)

# creating a confusion matrix
cm = confusion_matrix(y_test, dtree_predictions)

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

print("Accuracy:",metrics.accuracy_score(y_test, dtree_predictions))
'''
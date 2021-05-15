import pickle

import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# ML Libraries
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import matplotlib.pyplot as plt

# Global Parameters
stop_words = set(stopwords.words('english'))
# tweets = pd.read_csv('../datasets/sentiment_dataset.csv')
# x = tweets['text']


def remove_unwanted_cols(dataset, cols):
    for col in cols:
        del dataset[col]
    return dataset


def load_dataset(filename, cols):
    dataset = pd.read_csv(filename, encoding='latin-1')
    dataset.columns = cols
    return dataset


# Load dataset
dataset = load_dataset('../datasets/emotion_dataset.csv', ['Sl no', 'Tweets', 'Search key', 'Feeling', 'cleaned_text'])

# Remove unwanted columns from dataset
dataset = remove_unwanted_cols(dataset, ['Sl no', 'Search key', 'cleaned_text'])

dataset.groupby('Feeling').count().plot.bar(ylim=0)
plt.show()

#Preprocess data

stemmer = PorterStemmer()

dataset.Tweets = dataset['Tweets'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in stop_words]).lower())

vectorizer = TfidfVectorizer(min_df= 3, stop_words="english", sublinear_tf=True, norm='l2', ngram_range=(1, 2))
final_features = vectorizer.fit_transform(dataset.Tweets).toarray()
print(final_features.shape)

#first we split our dataset into testing and training set:
# this block is to split the dataset into training and testing set
X = np.array(dataset.iloc[:, 0]).ravel()

Y = np.array(dataset.iloc[:, 1]).ravel()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)


# instead of doing these steps one at a time, we can use a pipeline to complete them all at once
pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k=1200)),
                     ('clf', RandomForestClassifier())])

# fitting our model and save it in a pickle for later use
model = pipeline.fit(X_train, y_train)
with open('RandomForest.pickle', 'wb') as f:
    pickle.dump(model, f)

ytest = np.array(y_test)

# confusion matrix and classification report(precision, recall, F1-score)
print(classification_report(ytest, model.predict(X_test)))
print(confusion_matrix(ytest, model.predict(X_test)))


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

    ps = PorterStemmer()
    stemmed_words = [ps.stem(w) for w in filtered_words]
    #lemmatizer = WordNetLemmatizer()
    #lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in filtered_words]

    return " ".join(stemmed_words)


def load_dataset(filename, cols):
    dataset = pd.read_csv(filename, encoding='latin-1')
    dataset.columns = cols
    return dataset


def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector

# Load dataset
dataset = load_dataset('../datasets/emotion_dataset.csv', ['Sl no', 'Tweets', 'Search key', 'Feeling', 'cleaned_text'])
#Preprocess data

dataset.Tweets = dataset['Tweets'].apply(preprocess_tweet_text)


# Split dataset into Train, Test

# Same tf vector will be used for Testing sentiments on unseen trending data
tf_vector = get_feature_vector(np.array(dataset.iloc[:, 1]).ravel())
X = tf_vector.transform(np.array(dataset.iloc[:, 1]).ravel())
y = np.array(dataset.iloc[:, 3]).ravel()

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
pickle.dump(LR_model, open('emot_classifier.sav', 'wb'))




print(classification_report(y_test, LR_model.predict(X_test)))
print(confusion_matrix(y_test, LR_model.predict(X_test)))
'''
''''''
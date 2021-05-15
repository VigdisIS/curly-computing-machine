import glob
import pickle

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from nltk import PorterStemmer, re
from nltk.corpus import stopwords

from models.sentiment_classifier import load_dataset, preprocess_tweet_text

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

'''
# load the model from disk
tf_vector = pickle.load(open('../models/emot_vectorizer.sav', 'rb'))
LR_model = pickle.load(open('../models/emot_classifier.sav', 'rb'))
'''

model = pickle.load(open('../models/RandomForest.pickle', 'rb'))

def emotional_analysis(path):
    print(path)

    for tweetset in glob.glob(path):

        sphere = tweetset.split('\\')[1]

        corpus = ""

        path = tweetset + "/*.csv"

        for fname in glob.glob(path):
            print(fname)

            # Load dataset
            dataset = load_dataset(fname, ['username', 'text', 'cleaned_text'])

            dataset.text = dataset['text'].apply(lambda x: " ".join(
                [stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in stop_words]).lower())

            test = np.array(dataset.iloc[:, 1]).ravel()

            # Using Logistic Regression model for prediction
            test_prediction_lr = model.predict(test)
            test_prediction_prob = model.predict_proba(test)

            print(test_prediction_lr)
            print(test_prediction_prob)


            '''
            # Preprocess data
            dataset.text = dataset['text'].apply(preprocess_tweet_text)

            test_feature = tf_vector.transform(np.array(dataset.text).ravel())

            # Using Logistic Regression model for prediction
            test_prediction_lr = LR_model.predict(test_feature)
            test_prediction_prob = LR_model.predict_proba(test_feature)

            print(test_prediction_lr)
            print(test_prediction_prob)
            '''



emotional_analysis('../tweets/*')
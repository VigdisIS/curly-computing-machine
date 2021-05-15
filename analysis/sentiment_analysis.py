import glob
import pickle

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


from models.sentiment_classifier import clean_tweets

# load the model from disk
tf_vector = pickle.load(open('../models/sent_vectorizer.sav', 'rb'))
LR_model = pickle.load(open('../models/sent_classifier.sav', 'rb'))



def plot_sphere_sents(path):
    print(path)

    data = []
    pos_tweets = []
    neg_tweets = []

    for tweetset in glob.glob(path):

        pos_count = 0
        neg_count = 0

        fig_ = plt.figure()

        x_ = np.arange(5)

        ax_ = fig_.add_subplot(111)

        accounts = []
        data_ = []
        pos_tweets_ = []
        neg_tweets_ = []

        path = tweetset + "/*.csv"

        for fname in glob.glob(path):
            print(fname)

            tweets = pd.read_csv(fname)

            tweet_text = tweets['text']

            preprocessed_text = clean_tweets(tweet_text)

            test_feature = tf_vector.transform(np.array(preprocessed_text).ravel())

            # Using Logistic Regression model for prediction
            test_prediction_lr = LR_model.predict(test_feature)
            test_prediction_prob = LR_model.predict_proba(test_feature)

            occurrences = np.count_nonzero(test_prediction_lr == 4)
            occurrencesv2 = np.count_nonzero(test_prediction_lr == 0)

            pos_tweets_.append(occurrences)
            neg_tweets_.append(occurrencesv2)

            accounts.append(tweets['username'][1])

            pos_count += occurrences
            neg_count += occurrencesv2

        data_.append(pos_tweets_)
        data_.append(neg_tweets_)

        ax_.bar(x_ + 0.00, data_[0], color='mediumseagreen', width=0.5)
        ax_.bar(x_ + 0.3, data_[1], color='indianred', width=0.5)

        ax_.set_title(tweetset.split('\\')[1])
        ax_.set_ylabel('Tweets')
        ax_.set_xlabel('Accounts')
        ax_.set_xticks(x_ + 0.15)
        ax_.set_xticklabels(tuple(accounts))
        ax_.legend(labels=['positive', 'negative'])
        plt.show()

        pos_tweets.append(pos_count)
        neg_tweets.append(neg_count)


    data.append(pos_tweets)
    data.append(neg_tweets)

    print(data)

    fig = plt.figure()

    X = np.arange(5)

    ax = fig.add_subplot(111)

    ax.bar(X + 0.00, data[0], color='mediumseagreen', width=0.5)
    ax.bar(X + 0.3, data[1], color='indianred', width=0.5)

    print("Creating:")
    ax.set_title('Sentiment')
    ax.set_ylabel('Tweets')
    ax.set_xlabel('Communities')
    ax.set_xticks(X + 0.15)
    ax.set_xticklabels(('eSports', 'NSFW', 'politicians', 'scientists', 'streamers'))
    ax.legend(labels=['positive', 'negative'])
    plt.show()
    print("Done:")


plot_sphere_sents('../tweets/*')




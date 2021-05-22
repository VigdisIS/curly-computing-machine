# IMPORTS


import glob
import nltk
import unicodecsv
from textnets import Corpus, Textnet
from models.sentiment_classifier import load_dataset, preprocess_tweet_text
nltk.download('wordnet')


# METHOD


def most_common_words(path):
    """ Method to iterate through each sphere and accounts within each sphere to extract the most frequent words
            from each sphere and save them as a dataset """

    # Arrays to keep track of the spheres and their most frequent words
    spheres = []
    freq_words = []

    for tweetset in glob.glob(path):
        """ Iterates through each sphere, keeping track of tweets from all accounts in the sphere """

        # Name of sphere
        sphere = tweetset.split('/')[2]

        # String which will contain all tweets from sphere
        corpus = ""

        # All accounts (datasets) within each sphere
        path = tweetset + "/*.csv"

        for fname in glob.glob(path):
            """ Iterates through each account, extracting pre-processed tweets and appending them to the 
                    sphere-corpus """

            # Loads dataset
            dataset = load_dataset(fname, ['username', 'text', 'cleaned_text'])

            # Preprocesses tweets
            dataset.text = dataset['text'].apply(preprocess_tweet_text)

            # Appends tweets from account to sphere-corpus
            corpus += ' '.join(dataset.iloc[:, 1])

        # Tokenizes sphere-corpus
        all_words = nltk.tokenize.word_tokenize(corpus)

        # Applies NLTK's FreqDist function to corpus
        all_word_dist = nltk.FreqDist(all_words)

        # Extracts the 10 most common words from sphere-corpus
        most_common = all_word_dist.most_common(10)

        # Appends name of sphere to 'sphere' array
        spheres.append(sphere)

        # Appends the 10 most common words from corpus to the 'freq_words' array
        words = [w for w, _ in most_common]
        freq_words.append(words)

    # Creates a dataset containing the sphere names and their most common words
    with open('word_freq_corpus.csv', 'wb') as file:
        writer = unicodecsv.writer(file, delimiter=',', quotechar='"')

        # Writes header row to file
        header = ['sphere', 'text']
        writer.writerow(header)

        # Writes sphere with their most common words to file
        for i in range(len(spheres)):
            sphere = [spheres[i]]
            words = [' '.join(freq_words[i])]
            writer.writerow(sphere + words)


# CODE


# Calling the method to create most frequent words dataset
most_common_words('../tweets/*')

# Using the dataset as a corpus in the TextNets library
freq_corpus = Corpus.from_csv('word_freq_corpus.csv', label_col='sphere', doc_col='text')
tn = Textnet(freq_corpus.tokenized(), min_docs=1)

# Generating a TextNets clustered network using the corpus
tn.plot(label_term_nodes=True,
        label_doc_nodes=True,
        show_clusters=True)

# Generating a TextNets network of spheres
spheres = tn.project(node_type='doc')
spheres.plot(label_nodes=True)

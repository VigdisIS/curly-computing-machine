from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from clean_dataset import clean_tweets
import pandas as pd



tweets = pd.read_csv('./datasets/data.csv')

x = tweets['Tweets']

'''
cleaned_tweets = clean_tweets(x)

tweets['cleaned_text'] = cleaned_tweets

tweets.to_csv('./datasets/data.csv', index=False)
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
pickle.dump(tf, open('emotion_vectorizer.sav', 'wb'))
# pickling the model
pickle.dump(dt, open('emotion_classifier.sav', 'wb'))

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
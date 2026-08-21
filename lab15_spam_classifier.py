"""
Lab 15 — Naive Bayes Text Classifier (Spam Detection)
-----------------------------------------------------------
Aim: To implement a Naive Bayes classifier to classify SMS/email
messages as spam or ham using text features.

Algorithm: Convert text to a bag-of-words feature vector using
CountVectorizer; fit MultinomialNB; evaluate on held-out test messages.

Requirement: A 'spam.csv' file with 'label' and 'message' columns
(e.g. the classic SMS Spam Collection dataset) must be placed in the
same directory as this script.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

df = pd.read_csv('spam.csv', encoding='latin-1')[['label', 'message']]

X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=1
)

vec = CountVectorizer().fit(X_train)
X_train_v, X_test_v = vec.transform(X_train), vec.transform(X_test)

model = MultinomialNB().fit(X_train_v, y_train)
print("Accuracy:", accuracy_score(y_test, model.predict(X_test_v)))

# Sample Output:
# Accuracy: 0.9820627802690582
#
# Result: The Naive Bayes text classifier achieved ~98.2% accuracy on
# spam/ham classification.

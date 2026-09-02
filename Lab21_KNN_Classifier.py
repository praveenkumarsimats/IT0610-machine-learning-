"""
Lab 21 — K-Nearest Neighbour Classifier (Iris Dataset)
Aim: To implement the K-Nearest Neighbour algorithm to classify the Iris dataset and print
correct/wrong predictions.
Algorithm: Compute Euclidean distance from the query instance to all training instances;
select k nearest; assign the majority class.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
    test_size=0.3, random_state=1)

model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Result: KNN achieved ~97.8% accuracy in classifying the Iris dataset.

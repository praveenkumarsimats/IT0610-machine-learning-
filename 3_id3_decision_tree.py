"""
Program 3: ID3 Decision Tree Algorithm
Builds a decision tree from the PlayTennis dataset using information
gain (entropy reduction) and classifies a new sample.

Dataset: datasets/playtennis.csv
"""

import math
import pandas as pd


def entropy(labels):
    total = len(labels)
    counts = labels.value_counts()
    ent = 0.0
    for count in counts:
        p = count / total
        ent -= p * math.log2(p)
    return ent


def info_gain(data, attribute, target):
    total_entropy = entropy(data[target])
    values = data[attribute].unique()
    weighted_entropy = 0.0
    for v in values:
        subset = data[data[attribute] == v]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset[target])
    return total_entropy - weighted_entropy


def id3(data, attributes, target):
    labels = data[target]

    # Pure node
    if len(labels.unique()) == 1:
        return labels.iloc[0]

    # No attributes left -> majority class
    if not attributes:
        return labels.mode()[0]

    # Choose attribute with highest information gain
    gains = {attr: info_gain(data, attr, target) for attr in attributes}
    best_attr = max(gains, key=gains.get)

    tree = {best_attr: {}}
    remaining_attrs = [a for a in attributes if a != best_attr]

    for value in data[best_attr].unique():
        subset = data[data[best_attr] == value]
        if subset.empty:
            tree[best_attr][value] = labels.mode()[0]
        else:
            tree[best_attr][value] = id3(subset, remaining_attrs, target)

    return tree


def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + "-> " + str(tree))
        return
    for attr, branches in tree.items():
        for value, subtree in branches.items():
            print(f"{indent}[{attr} = {value}]")
            print_tree(subtree, indent + "    ")


def classify(tree, sample):
    if not isinstance(tree, dict):
        return tree
    attr = next(iter(tree))
    value = sample.get(attr)
    subtree = tree[attr].get(value)
    if subtree is None:
        return "Unknown"
    return classify(subtree, sample)


if __name__ == "__main__":
    data = pd.read_csv("datasets/playtennis.csv")
    target = "PlayTennis"
    attributes = [c for c in data.columns if c != target]

    tree = id3(data, attributes, target)

    print("Learned Decision Tree:\n")
    print_tree(tree)

    new_sample = {
        "Outlook": "Sunny",
        "Temperature": "Cool",
        "Humidity": "High",
        "Wind": "Strong",
    }
    prediction = classify(tree, new_sample)
    print(f"\nNew sample: {new_sample}")
    print(f"Predicted class: {prediction}")

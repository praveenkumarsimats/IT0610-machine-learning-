"""
Program 1: FIND-S Algorithm
Finds the most specific hypothesis consistent with the given positive
training examples read from a CSV file.

Dataset: datasets/enjoysport.csv
Last column is the target concept (Yes/No). Only positive ("Yes")
examples are used by FIND-S.
"""

import csv


def load_data(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [row for row in reader]
    return header, data


def find_s(header, data):
    attributes = header[:-1]
    n_attr = len(attributes)

    # Step 1: initialize hypothesis to the most specific one
    hypothesis = ["0"] * n_attr

    print("Attributes:", attributes)
    print("\nStep-by-step hypothesis updates:\n")

    first_positive_found = False
    for i, row in enumerate(data, start=1):
        *attrs, label = row
        if label.strip().lower() != "yes":
            print(f"Example {i}: {row} -> Negative, ignored")
            continue

        if not first_positive_found:
            hypothesis = attrs.copy()
            first_positive_found = True
        else:
            for j in range(n_attr):
                if hypothesis[j] != attrs[j]:
                    hypothesis[j] = "?"

        print(f"Example {i}: {row} -> Positive, hypothesis = {hypothesis}")

    return hypothesis


if __name__ == "__main__":
    header, data = load_data("datasets/enjoysport.csv")
    final_hypothesis = find_s(header, data)
    print("\nFinal maximally specific hypothesis (FIND-S):")
    print(final_hypothesis)

"""
Program 2: Candidate-Elimination Algorithm
Outputs the version space (set of all hypotheses consistent with the
training examples) using the S (specific) and G (general) boundaries.

Dataset: datasets/enjoysport.csv
"""

import csv


def load_data(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [row for row in reader]
    return header, data


def more_general(h1, h2):
    """Return True if h1 is more general than or equal to h2."""
    for x, y in zip(h1, h2):
        if x != "?" and (x != y and y != "?"):
            return False
    return True


def is_consistent(hypothesis, instance):
    return all(h == "?" or h == a for h, a in zip(hypothesis, instance))


def candidate_elimination(header, data):
    n_attr = len(header) - 1
    S = ["0"] * n_attr                 # most specific boundary
    G = [["?"] * n_attr]               # most general boundary set

    print(f"Initial S: {S}")
    print(f"Initial G: {G}\n")

    for i, row in enumerate(data, start=1):
        *attrs, label = row
        positive = label.strip().lower() == "yes"

        if positive:
            # Remove from G any hypothesis inconsistent with attrs
            G = [g for g in G if is_consistent(g, attrs)]

            # Generalize S minimally to cover attrs
            if S == ["0"] * n_attr:
                S = attrs.copy()
            else:
                for j in range(n_attr):
                    if S[j] != attrs[j]:
                        S[j] = "?"

        else:
            # Specialize G to exclude attrs, keeping only hyps still
            # more general than S
            new_G = []
            for g in G:
                if is_consistent(g, attrs):
                    # g wrongly covers the negative example, specialize it
                    for j in range(n_attr):
                        if g[j] == "?":
                            if S[j] not in ("0", "?") and S[j] != attrs[j]:
                                specialized = g.copy()
                                specialized[j] = S[j]
                                if more_general(specialized, S) or specialized == S:
                                    new_G.append(specialized)
                else:
                    new_G.append(g)
            # remove hypotheses in new_G that are more specific than another
            G = []
            for g in new_G:
                if g not in G:
                    G.append(g)

        print(f"After example {i} ({row}):")
        print(f"  S: {S}")
        print(f"  G: {G}\n")

    return S, G


if __name__ == "__main__":
    header, data = load_data("datasets/enjoysport.csv")
    S_final, G_final = candidate_elimination(header, data)
    print("Final Specific hypothesis (S):", S_final)
    print("Final General hypotheses (G):", G_final)

"""
Lab 26 — Case-Based Reasoning (Simple CBR System)
Aim: To implement a simple Case-Based Reasoning system that retrieves the most similar
past case and reuses its solution for a new problem.
Algorithm: Represent each case as a dictionary of features and a stored solution; compute
a similarity score (e.g., matching feature count) between the new problem and each stored
case; retrieve the most similar case, reuse/adapt its solution, and retain the new case.
"""

case_library = [
    {"symptoms": {"fever": True, "cough": True, "fatigue": False}, "diagnosis": "Flu"},
    {"symptoms": {"fever": False, "cough": True, "fatigue": True}, "diagnosis": "Cold"},
    {"symptoms": {"fever": True, "cough": False, "fatigue": True}, "diagnosis": "Viral Infection"},
]


def similarity(a, b):
    return sum(1 for k in a if a[k] == b.get(k))


def retrieve(new_case):
    scored = [(similarity(new_case, c["symptoms"]), c) for c in case_library]
    return max(scored, key=lambda x: x[0])[1]


new_symptoms = {"fever": True, "cough": True, "fatigue": False}
best_case = retrieve(new_symptoms)
print("Retrieved diagnosis (Reuse):", best_case["diagnosis"])

case_library.append({"symptoms": new_symptoms, "diagnosis": best_case["diagnosis"]})  # Retain
print("Case library size after Retain:", len(case_library))

# Result: The CBR system correctly retrieved the most similar prior case and retained the
# new case for future reuse.

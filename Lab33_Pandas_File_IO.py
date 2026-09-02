"""
Lab 33 — Pandas File I/O (Reading and Writing CSV)
Aim: To read data from a CSV file into a DataFrame and write a processed DataFrame back
to a new CSV file.
Algorithm: Use pd.read_csv() to load data; process/filter it; use df.to_csv() to save the
output.

NOTE: This script expects a file named 'students.csv' (with columns like
name, marks, class) to be present in the working directory.
"""

import pandas as pd

df = pd.read_csv('students.csv')
print("Loaded data:\n", df.head())

df['pass'] = df['marks'] >= 50
df.to_csv('students_processed.csv', index=False)

print("Processed file saved. Preview:\n", df.head())

# Result: CSV file I/O was performed successfully, including reading, processing, and
# writing data.

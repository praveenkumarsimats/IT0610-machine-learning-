"""
Lab 32 — Pandas DataFrame Creation and Combining
Aim: To create Pandas DataFrames from dictionaries and combine them using
concatenation and merging.
Algorithm: Construct DataFrames from Python dictionaries; use pd.concat() to stack
rows and pd.merge() to join on a common key.
"""

import pandas as pd

df1 = pd.DataFrame({'name': ['A', 'B', 'C'], 'marks': [80, 90, 70]})
df2 = pd.DataFrame({'name': ['D'], 'marks': [85]})

combined = pd.concat([df1, df2], ignore_index=True)
print("Concatenated:\n", combined)

df3 = pd.DataFrame({'name': ['A', 'B', 'C'], 'grade': ['A', 'A+', 'B']})
merged = pd.merge(df1, df3, on='name', how='inner')
print("Merged:\n", merged)

# Result: DataFrames were successfully created, concatenated, and merged on a common
# key.

"""
Lab 35 — Pandas Filtering and Sorting
Aim: To filter DataFrame rows based on conditions and sort the results by one or more
columns.
Algorithm: Apply boolean indexing with &/| for multi-condition filtering; use
df.sort_values() for single/multi-column sorting.
"""

import pandas as pd

df = pd.DataFrame({'name': ['A', 'B', 'C', 'D'],
                    'class': ['I', 'II', 'I', 'II'],
                    'marks': [78, 85, 92, 66]})

filtered = df[(df['marks'] > 70) & (df['class'] == 'I')]
print("Filtered:\n", filtered)

sorted_df = df.sort_values(['class', 'marks'], ascending=[True, False])
print("Sorted:\n", sorted_df)

# Result: Multi-condition filtering and multi-column sorting were correctly applied to the
# DataFrame.

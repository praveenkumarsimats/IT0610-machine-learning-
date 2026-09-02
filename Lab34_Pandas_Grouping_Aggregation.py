"""
Lab 34 — Pandas Grouping and Aggregation
Aim: To group data by a categorical column and compute aggregate statistics for each
group.
Algorithm: Use df.groupby(col) followed by aggregation functions such as .mean(),
.sum(), or .agg() with multiple functions.
"""

import pandas as pd

df = pd.DataFrame({'class': ['I', 'II', 'I', 'II', 'I', 'II'],
                    'marks': [78, 85, 92, 66, 74, 88]})

print("Mean marks per class:\n", df.groupby('class')['marks'].mean())
print("Multiple aggregates:\n", df.groupby('class')['marks'].agg(['mean', 'max', 'min']))

# Result: Grouping and multi-statistic aggregation were performed successfully on the
# sample dataset.

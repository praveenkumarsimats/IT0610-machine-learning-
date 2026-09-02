"""
Lab 37 — Data Cleaning with Pandas (Missing Values & Duplicates)
Aim: To identify and handle missing values and duplicate records in a dataset using
Pandas.
Algorithm: Use isnull().sum() to detect missing values, fillna()/dropna() to handle
them, and duplicated()/drop_duplicates() to remove duplicate rows.
"""

import pandas as pd
import numpy as np

df = pd.DataFrame({'name': ['A', 'B', 'B', 'C', None],
                    'marks': [78, np.nan, 85, 92, 66]})

print("Missing values per column:\n", df.isnull().sum())

df_filled = df.fillna({'name': 'Unknown', 'marks': df['marks'].mean()})
print("After filling missing values:\n", df_filled)

df_clean = df_filled.drop_duplicates()
print("After removing duplicates:\n", df_clean)

# Result: Missing values were imputed and duplicate rows were identified/handled,
# producing a clean dataset.

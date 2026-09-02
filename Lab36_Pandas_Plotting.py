"""
Lab 36 — Pandas Plotting (Data Visualisation)
Aim: To visualise a DataFrame's data using Pandas' built-in plotting functions (bar chart
and histogram).
Algorithm: Use df.plot(kind=...) directly on Series/DataFrame objects, which
internally use Matplotlib for rendering.
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({'name': ['A', 'B', 'C', 'D', 'E'], 'marks': [78, 85, 92, 66, 74]})

df['marks'].plot(kind='hist', title='Marks Distribution', bins=5)
plt.savefig('marks_hist.png')
plt.clf()

df.plot(x='name', y='marks', kind='bar', title='Marks by Student', legend=False)
plt.savefig('marks_bar.png')

print("Plots saved: marks_hist.png, marks_bar.png")

# Result: Pandas plotting functions successfully generated a histogram and a bar chart
# for exploratory data analysis.

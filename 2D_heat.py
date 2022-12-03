import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('.\Measurement_Data\\2D_heat\\3 deg\Record_2022-11-29_09-31-06.csv',delimiter=';')
ax = sns.heatmap(df,cmap='cubehelix')
print(df)
plt.show()
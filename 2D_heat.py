import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df1 = pd.read_csv('.\Measurement_Data\\2D_heat\\3 deg\\1.csv',delimiter=';', skipinitialspace=True)
df1 = df1.dropna(axis=1)
df1 = df1.astype(float)
df2 = pd.read_csv('.\Measurement_Data\\2D_heat\\3 deg\\2.csv',delimiter=';', skipinitialspace=True)
df2 = df2.dropna(axis=1)
df2 = df2.astype(float)
df3 = pd.read_csv('.\Measurement_Data\\2D_heat\\3 deg\\3.csv',delimiter=';', skipinitialspace=True)
df3 = df3.iloc[: , :-1]
df4 = pd.read_csv('.\Measurement_Data\\2D_heat\\3 deg\\4.csv',delimiter=';', skipinitialspace=True)
df4 = df4.iloc[: , :-1]
df = df1.add(df2, fill_value=0)
df = df.apply(lambda x: x/2)
#df = df.add(df3, fill_value=0)
#df = df.add(df4, fill_value=0)
print(df)
df = df1
ma = df.values.max()
mi = df.values.min()
df = df.apply(lambda x: (x-mi)/(ma-mi))
df = df.apply(lambda x: np.tanh(2*(x-0.8)))
a = sns.set_palette(sns.color_palette(['#000000','#ffffff'], as_cmap=True))
a = sns.color_palette('Greys_r', 1000)
ax = sns.heatmap(df,cmap=a, yticklabels=False, xticklabels=False)
print(df.iloc[225,220])
plt.show()
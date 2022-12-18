import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_csv('.\Measurement_Data\\2D_heat\\3 deg\Record_2022-11-29_09-31-06.csv',delimiter=';', skipinitialspace=True)
df = df.iloc[: , :-1]
ma = df.values.max()
mi = df.values.min()
df = df.apply(lambda x: (x-mi)/(ma-mi))
df = df.apply(lambda x: np.tanh(2*(x-0.73)))
a = sns.set_palette(sns.color_palette(['#000000','#ffffff'], as_cmap=True))
a = sns.color_palette('Greys_r', 1000)
ax = sns.heatmap(df,cmap=a, yticklabels=False, xticklabels=False)
print(df.iloc[225,220])
plt.show()
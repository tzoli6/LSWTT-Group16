import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme()

df = pd.read_csv('G:\My Drive\TU Delft\LowSpeedWindTunnelTest\LSWTT-Group16\Measurement_Data\\twoD\main.csv',skipinitialspace=True)
#df=df.drop(labels=0)
print(df)
sns.scatterplot(data=df, x='Alpha' ,y='Cl')
sns.relplot(data=df, x='Alpha' ,y='Cl',kind='line')

plt.show()
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme()

df = pd.read_csv('.\Measurement_Data\\twoD\cp_data.csv',skipinitialspace=True)
#df = pd.read_csv('G:\My Drive\TU Delft\LowSpeedWindTunnelTest\LSWTT-Group16\Measurement_Data\\twoD\cp_data.csv',skipinitialspace=True)
df = df.T
print(df)
print(df[1])
for i in range(1,43):
    #sns.scatterplot(data=df, x='Alpha' ,y='Cl')
    #sns.relplot(data=df, x='Alpha' ,y='Cl',kind='line')
    sns.scatterplot(data=df, x=df[0], y=df[i])
    plt.plot(df[0], df[i])
    plt.savefig('.\Measurement_Data\\twoD\Graphs\\'+str(df[i]['Alpha'])+'.jpg')
    #plt.savefig('G:\My Drive\TU Delft\LowSpeedWindTunnelTest\LSWTT-Group16\Measurement_Data\\twoD\Graphs\\'+str(df[i]['Alpha'])+'.jpg')
    plt.clf()
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme()
sns.set_style('darkgrid')
df = pd.read_csv('.\Measurement_Data\\twoD\cp_data.csv',skipinitialspace=True)
#df = pd.read_csv('G:\My Drive\TU Delft\LowSpeedWindTunnelTest\LSWTT-Group16\Measurement_Data\\twoD\cp_data.csv',skipinitialspace=True)
df = df.T
print(df)
print(df[28])
for i in range(28,29):
    #sns.scatterplot(data=df, x='Alpha' ,y='Cl')
    #sns.relplot(data=df, x='Alpha' ,y='Cl',kind='line')
    plt.plot(df[0]/100, df[i], color='grey',linewidth='1')
    f=sns.scatterplot(data=df, x=df[0]/100, y=df[i], marker ='X', color='seagreen', legend=False)
    f.set(xlabel='x/c [-]', ylabel='Cp [-]')
    plt.legend()
    #f.invert_yaxis()
    plt.savefig('.\Measurement_Data\\twoD\Graphs\\'+str(df[i]['Alpha'])+'.svg',dpi=1200)
    #plt.savefig('G:\My Drive\TU Delft\LowSpeedWindTunnelTest\LSWTT-Group16\Measurement_Data\\twoD\Graphs\\'+str(df[i]['Alpha'])+'.svg',format='svg', dpi=1200)
    plt.clf()
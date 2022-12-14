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
print(df[1])
for i in range(1,43,6):
    #sns.scatterplot(data=df, x='Alpha' ,y='Cl')
    #sns.relplot(data=df, x='Alpha' ,y='Cl',kind='line')
    plt.plot(df[0]/100, df[i], color='grey',linewidth='1')
    f=sns.scatterplot(data=df, x=df[0]/100, y=df[i])
    f.set(xlabel='x/c [-]', ylabel='Cp [-]')
    f.invert_yaxis()
    f.legend(str(df[i][0]))
    plt.savefig('.\Measurement_Data\\twoD\Graphs\\'+'all_cp'+'.svg',dpi=1200)
    #plt.savefig('G:\My Drive\TU Delft\LowSpeedWindTunnelTest\LSWTT-Group16\Measurement_Data\\twoD\Graphs\\'+str(df[i]['Alpha'])+'.svg',format='svg', dpi=1200)
    #plt.clf()
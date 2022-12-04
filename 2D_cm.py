import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme()
sns.set_style('darkgrid')
df = pd.read_csv('.\Measurement_Data\\twoD\main.csv',skipinitialspace=True)
#df = pd.read_csv('G:\My Drive\TU Delft\LowSpeedWindTunnelTest\LSWTT-Group16\Measurement_Data\\twoD\cp_data.csv',skipinitialspace=True)
df = pd.DataFrame(df)
print(df)
df = df[['Alpha','Cm']]
plt.plot(df['Alpha'], df['Cm'], color='grey',linewidth='1')
f=sns.scatterplot(data=df, x=df['Alpha'], y=df['Cm'], marker ='X', color='seagreen')
f.set(xlabel='Alpha [degrees]', ylabel='Cm [-]')
plt.savefig('.\Measurement_Data\\twoD\Graphs\\'+str(['Cm-Alpha'])+'.svg',dpi=1200)
    #plt.savefig('G:\My Drive\TU Delft\LowSpeedWindTunnelTest\LSWTT-Group16\Measurement_Data\\twoD\Graphs\\'+str(df[i]['Alpha'])+'.svg',format='svg', dpi=1200)
plt.clf()
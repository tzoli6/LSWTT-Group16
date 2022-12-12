from cProfile import label
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
df = df[['Alpha','Cl','Cm','Cd']]
da = df[:33]
print(df)
plt.plot(df['Alpha'], df['Cl'], color='red',linewidth='1', label = 'Cl')
plt.plot(df['Alpha'], df['Cm'], color='green',linewidth='1', label = 'Cm')
plt.legend
f=sns.scatterplot(data=df, x=df['Alpha'], y=df['Cl'], marker ='X', color='red')
f=sns.scatterplot(data=df, x=df['Alpha'], y=df['Cm'], marker ='s', color='seagreen')
f.set(xlabel='Alpha [degrees]', ylabel='Cl/Cm [-]')
plt.savefig('.\Measurement_Data\\twoD\Graphs\\'+str(['ClCm-Alpha'])+'.svg',dpi=1200)
    #plt.savefig('G:\My Drive\TU Delft\LowSpeedWindTunnelTest\LSWTT-Group16\Measurement_Data\\twoD\Graphs\\'+str(df[i]['Alpha'])+'.svg',format='svg', dpi=1200)
plt.clf()

plt.plot(da['Alpha'], da['Cl'], color='red',linewidth='1',label = 'Cl')
plt.plot(da['Alpha'], da['Cm'], color='green',linewidth='1',label = 'Cm')
plt.legend
q=sns.scatterplot(data=da, x=da['Alpha'], y=da['Cl'], marker ='X', color='red')
f=sns.scatterplot(data=da, x=da['Alpha'], y=da['Cm'], marker ='s', color='seagreen')
q.set(xlabel='Alpha [degrees]', ylabel='Cl/Cm [-]')
plt.savefig('.\Measurement_Data\\twoD\Graphs\\'+str(['ClCm-Alpha_nohysteresis'])+'.svg',dpi=1200)
    #plt.savefig('G:\My Drive\TU Delft\LowSpeedWindTunnelTest\LSWTT-Group16\Measurement_Data\\twoD\Graphs\\'+str(df[i]['Alpha'])+'.svg',format='svg', dpi=1200)
plt.clf()

plt.plot(df['Cl'], df['Cd'], color='gray',linewidth='1')
plt.legend
f=sns.scatterplot(data=df, x=df['Cl'], y=df['Cd'], marker ='X', color='seagreen')
f.set(xlabel='Cl [-]', ylabel='Cd [-]')
plt.savefig('.\Measurement_Data\\twoD\Graphs\\'+str(['Cl-Cd'])+'.svg',dpi=1200)
    #plt.savefig('G:\My Drive\TU Delft\LowSpeedWindTunnelTest\LSWTT-Group16\Measurement_Data\\twoD\Graphs\\'+str(df[i]['Alpha'])+'.svg',format='svg', dpi=1200)
plt.clf()

plt.plot(da['Cl'], da['Cd'], color='gray',linewidth='1')
plt.legend
q=sns.scatterplot(data=da, x=da['Cl'], y=da['Cd'], marker ='X', color='seagreen')
q.set(xlabel='Cl [-]', ylabel='Cd [-]')
plt.savefig('.\Measurement_Data\\twoD\Graphs\\'+str(['Cl-Cd_nohysteresis'])+'.svg',dpi=1200)
    #plt.savefig('G:\My Drive\TU Delft\LowSpeedWindTunnelTest\LSWTT-Group16\Measurement_Data\\twoD\Graphs\\'+str(df[i]['Alpha'])+'.svg',format='svg', dpi=1200)
plt.clf()

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
df = df[['alpha','CL','Cm','CD']]
da = df[:33]
print(df)
plt.plot(df['alpha'], df['CL'],linewidth='1', label = r'$C_l$', color='grey')
plt.plot(df['alpha'], df['Cm'], linewidth='1', label = r'$C_m$', color='grey')
plt.legend
f=sns.scatterplot(data=df, x=df['alpha'], y=df['CL'], marker ='X', color='red')
f=sns.scatterplot(data=df, x=df['alpha'], y=df['Cm'], marker ='s', color='seagreen')
f.set(xlabel=r'$\alpha$ [$^\circ$]', ylabel=r'$C_l/C_m$ [-]')
plt.savefig('.\Measurement_Data\\twoD\Graphs\\'+str('ClCm-Alpha')+'.svg',dpi=1200)
    #plt.savefig('G:\My Drive\TU Delft\LowSpeedWindTunnelTest\LSWTT-Group16\Measurement_Data\\twoD\Graphs\\'+str(df[i]['Alpha'])+'.svg',format='svg', dpi=1200)
plt.clf()

plt.plot(da['alpha'], da['CL'], linewidth='1',label =  r'$C_l$', color='grey')
plt.plot(da['alpha'], da['Cm'], linewidth='1',label = r'$C_m$', color='grey')
plt.legend
q=sns.scatterplot(data=da, x=da['alpha'], y=da['CL'], marker ='X', color='red')
f=sns.scatterplot(data=da, x=da['alpha'], y=da['Cm'], marker ='s', color='seagreen')
q.set(xlabel=r'$\alpha$ [$^\circ$]', ylabel=r'$C_l/C_m$ [-]')
plt.savefig('.\Measurement_Data\\twoD\Graphs\\'+str('ClCm-Alpha_nohysteresis')+'.svg',dpi=1200)
    #plt.savefig('G:\My Drive\TU Delft\LowSpeedWindTunnelTest\LSWTT-Group16\Measurement_Data\\twoD\Graphs\\'+str(df[i]['Alpha'])+'.svg',format='svg', dpi=1200)
plt.clf()

plt.plot(df['CL'], df['CD'], color='gray',linewidth='1')
plt.legend
f=sns.scatterplot(data=df, x=df['CL'], y=df['CD'], marker ='X', color='seagreen')
f.set(xlabel=r'$C_l$ [-]', ylabel=r'$C_d$ [-]')
plt.savefig('.\Measurement_Data\\twoD\Graphs\\'+str(['Cl-Cd'])+'.svg',dpi=1200)
    #plt.savefig('G:\My Drive\TU Delft\LowSpeedWindTunnelTest\LSWTT-Group16\Measurement_Data\\twoD\Graphs\\'+str(df[i]['Alpha'])+'.svg',format='svg', dpi=1200)
plt.clf()

plt.plot(da['CL'], da['CD'], color='gray',linewidth='1')
plt.legend
q=sns.scatterplot(data=da, x=da['CL'], y=da['CD'], marker ='X', color='seagreen')
q.set(xlabel=r'$C_l$ [-]', ylabel=r'$C_d$ [-]')
plt.savefig('.\Measurement_Data\\twoD\Graphs\\'+str(['Cl-Cd_nohysteresis'])+'.svg',dpi=1200)
    #plt.savefig('G:\My Drive\TU Delft\LowSpeedWindTunnelTest\LSWTT-Group16\Measurement_Data\\twoD\Graphs\\'+str(df[i]['Alpha'])+'.svg',format='svg', dpi=1200)
plt.clf()

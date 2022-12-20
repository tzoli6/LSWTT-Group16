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
for i in range(34,44):
    #sns.scatterplot(data=df, x='Alpha' ,y='Cl')
    #sns.relplot(data=df, x='Alpha' ,y='Cl',kind='line')
    plt.plot(df[0]/100, df[i], color='grey',linewidth='1')
    f=sns.scatterplot(data=df, x=df[0]/100, y=df[i], marker ='X', color='seagreen', legend=False)
    f.set(xlabel='x/c [-]', ylabel='Cp [-]')
    #plt.legend()
    f.invert_yaxis()
    plt.savefig('.\Measurement_Data\\twoD\Graphs_corr\\'+str(df[i]['Alpha'])+'hyst.svg',dpi=1200)
    #plt.savefig('G:\My Drive\TU Delft\LowSpeedWindTunnelTest\LSWTT-Group16\Measurement_Data\\twoD\Graphs\\'+str(df[i]['Alpha'])+'.svg',format='svg', dpi=1200)
    plt.clf()

y = [-0.37209,-0.41389,-0.45902,-0.45665,-0.45861,-0.46265,-0.46977,-0.47431,-0.47225,-0.46738,-0.45719,-0.45094,-0.44791,-0.4643,-0.52515,-0.65409,-0.86037,-1.13822,-1.52581,-1.68166,-1.8824,-1.98788,-2.47731,-2.81404,-4.92138,-5.24406,-3.87034,-3.87444,0.34385,0.89223,1,0.96666,0.91763,0.84047,0.77672,0.69135,0.54177,0.40473,0.2839,0.19316,0.10055,0.02313,-0.02268,-0.04925,-0.06555,-0.08113,-0.08721,-0.08855,-0.09107,-0.10684,-0.1304,-0.14251,-0.22363,-0.37209]
x = [100,95.2,90.1,85.2,80.1,75,70.2,65,59.8,55.2,50.1,45.2,40.2,35.1,30.1,25,20,15.1,10.1,8.9,7.1,5.5,4.3,3.1,1.8,0.8,0,0,0.6,1.6,3,4.2,5.4,6.9,8.4,9.8,15,19.9,24.9,29.9,34.9,40,45,49.6,55.1,60.1,65,69.9,75,80,85,90,95.2,100]

plt.plot(x, y)
plt.show()
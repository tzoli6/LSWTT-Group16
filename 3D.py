import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme()

df = pd.read_csv('.\Measurement_Data\\3D\corr_test.csv',skipinitialspace=True)
df = pd.DataFrame(df)

#There were some datapoints which are redundant. These are the following:
#Run-nr
#29
#32
#33
#34

labels=[29-1,32-1,33-1,34-1]
df = df.drop(labels)
print(df)

###CL-Alpha###
df_1 = df[['Alpha','CL']]
print(df_1)
#plot
plt.plot(df_1['Alpha'], df_1['CL'], color='grey',linewidth='1')
f=sns.scatterplot(data=df_1, x=df_1['Alpha'], y=df_1['CL'], marker ='X', color='seagreen')
f.set(xlabel='Alpha [degrees]', ylabel='CL [-]')
plt.show()

###CL-CD###

df_2 = df[['CD','CL']]
print(df_2)
#plot
plt.plot(df_2['CD'], df_2['CL'], color='grey',linewidth='1')
f=sns.scatterplot(data=df_2, x=df_2['CD'], y=df_2['CL'], marker ='X', color='seagreen')
f.set(xlabel='CD [-]', ylabel='CL [-]')
plt.show()


###CD-Alpha###

df_3 = df[['Alpha','CD']]
print(df_3)
#plot
plt.plot(df_3['Alpha'], df_3['CD'], color='grey',linewidth='1')
f=sns.scatterplot(data=df_3, x=df_3['Alpha'], y=df_3['CD'], marker ='X', color='seagreen')
f.set(xlabel='Alpha [degrees]', ylabel='CD [-]')
plt.show()

###CM-Alpha###

df_4 = df[['Alpha','Cm_pitch']]
print(df_4)
#plot
plt.plot(df_4['Alpha'], df_4['Cm_pitch'], color='grey',linewidth='1')
f=sns.scatterplot(data=df_4, x=df_4['Alpha'], y=df_4['Cm_pitch'], marker ='X', color='seagreen')
f.set(xlabel='Alpha [degrees]', ylabel='Cm [-]')
plt.show()

#sns.relplot(data=df, x='Alpha' ,y='CL',markers='O')
#plt.plot(df['Alpha-pr'],df['Cl-press'])
#plt.show()
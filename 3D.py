import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme()

###CL-Alpha###

df = pd.read_csv('.\Measurement_Data\\3D\corr_test.csv',skipinitialspace=True)
df = pd.DataFrame(df)
df_1 = df[['Alpha','CL']]
print(df_1)

###CL-CD###

df_2 = df[['CD','CL']]
print(df_2)


###CD-Alpha###

df_3 = df[['Alpha','CD']]
print(df_3)

###CM-Alpha###

df_4 = df[['Alpha','Cm_pitch']]
print(df_4)

#sns.relplot(data=df, x='Alpha' ,y='CL',markers='O')
#plt.plot(df['Alpha-pr'],df['Cl-press'])
#plt.show()
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme()

df = pd.read_csv('.\Measurement_Data\\3D\corr_test.csv',skipinitialspace=True)
df = pd.DataFrame(df)
df = df[['Alpha','Cl']]
print(df)

#sns.relplot(data=df, x='Alpha' ,y='CL',markers='O')
#plt.plot(df['Alpha-pr'],df['Cl-press'])
#plt.show()
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

sns.set_theme()
sns.set_style('darkgrid')


df = pd.read_csv(".\CFD\\3D\\3Dwindtunnelmodel_a=0.00_v=49.00ms_induced_angle.csv", skipinitialspace=True, delimiter=';')

data = pd.DataFrame(df)
data_vector = data[['y-span', 'Ai']]
plt.plot(data_vector['y-span'], data_vector['Ai'], color='grey',linewidth='1')

f = sns.relplot(data=df, x='y-span', y='Ai', markers='o')
f.set(xlabel='y-span [m]', ylabel='Ai [degrees]')
plt.show()
print(df)
plt.savefig('.\CFD\\3D\graphs\\'+str(['induced_angle-alpha=0'])+'.svg',dpi=1200)
plt.clf()

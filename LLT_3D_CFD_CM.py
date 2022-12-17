import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

sns.set_theme()

df = pd.read_csv(".\CFD\\3D\\3D_windtunnel_model_T1-49_0_m_s-LLT_V2_(3).csv", skipinitialspace=True)

f = sns.relplot(data=df, x='alpha', y='Cm', markers='o')
data = pd.DataFrame(df)
data_vector = data[['alpha', 'Cm']]
plt.plot(data_vector['alpha'], data_vector['Cm'], color='grey', linewidth='1')
f.set(xlabel='alpha [-]', ylabel='Cm [-]')
plt.show()
plt.savefig('.\CFD\\3D\graphs\\'+str(['Cm-alpha'])+'.svg',dpi=1200)
plt.clf()

import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

sns.set_theme()

df = pd.read_csv(".\CFD\\3D\\3D_windtunnel_model_T1-49_0_m_s-LLT_V2_(3).csv", skipinitialspace=True)

f = sns.relplot(data=df, x='CD', y='CL', markers='o', color = 'purple', aspect=16/11)
data = pd.DataFrame(df)
data_vector = data[['CD', 'CL']]
plt.plot(data_vector['CD'], data_vector['CL'], color='grey', linewidth='1')
f.set(xlabel='CD [-]', ylabel='CL [-]')

plt.savefig('.\CFD\\3D\graphs\\LLT_CFD_CL-CD.svg',dpi=1200)
plt.show()
plt.clf()

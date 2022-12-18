import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

sns.set_theme()

#df = pd.read_csv(".\CFD\\3D\\3D_windtunnel_model_T1-49_0_m_s-LLT_V2.csv", skipinitialspace=True, delimiter=";")
# we reran the simulations for LLT but with stepsize of 0.5 degrees

df = pd.read_csv(".\CFD\\3D\\3D_windtunnel_model_T1-49_0_m_s-LLT_V2_(3).csv", skipinitialspace=True)

f = sns.relplot(data=df, x= 'alpha', y='CL', markers='o')
data = pd.DataFrame(df)
data_vector = data[['alpha', 'CL']]
plt.plot(data_vector['alpha'], data_vector['CL'], color='grey', linewidth='1')
f.set(xlabel='alpha [degrees]', ylabel='CL [-]')

print(df)
plt.savefig('.\CFD\\3D\graphs\\'+str(['CL-alpha'])+'.svg',dpi=1200)
plt.show()
plt.clf()

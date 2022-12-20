import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

sns.set_theme()

df = pd.read_csv(".\CFD\\3D\\3D_windtunnel_model_T1-49_0_m_s-Panel.csv", skipinitialspace=True)

f = sns.relplot(data=df, x='alpha', y='CL', markers='o')
data = pd.DataFrame(df)
data_vector = data[['alpha', 'CL']]
plt.plot(data_vector['alpha'], data_vector['CL'], color='grey', linewidth='1')
f.set(xlabel='\u03B1 [\u00B0]', ylabel='CL [-]')

print(df)
plt.savefig('.\CFD\\3D\graphs\\'+str(['Panel_CL-alpha'])+'.svg',dpi=1200)
plt.show()
plt.clf()


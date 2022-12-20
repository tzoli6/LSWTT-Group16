import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

sns.set_theme()

df = pd.read_csv(".\CFD\\3D\\3D_windtunnel_model_T1-49_0_m_s-LLT_V2_(3).csv", skipinitialspace=True)

f = sns.relplot(data=df, x='alpha', y='CDi', markers='o', color = 'purple', aspect=16/11)

data = pd.DataFrame(df)
data_vector = data[['alpha', 'CDi']]
plt.plot(data_vector['alpha'], data_vector['CDi'], color='grey', linewidth='1')
f.set(xlabel='alpha [degrees]', ylabel='CDi [-]')
f.set(xlabel='\u03B1 [\u00B0]', ylabel='CDi [-]')
plt.savefig('.\CFD\\3D\graphs\\LLT_cfd_CDi-alpha.svg',dpi=1200)
plt.show()
plt.clf()

import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

sns.set_theme()

#df = pd.read_csv(".\CFD\\3D\\3D_windtunnel_model_T1-49_0_m_s-LLT_V2.csv", skipinitialspace=True, delimiter=";")
# we reran the simulations for LLT but with stepsize of 0.5 degrees

df = pd.read_csv(".\CFD\\3D\\3D_windtunnel_model_T1-49_0_m_s-LLT_V2_(3).csv", skipinitialspace=True)

sns.relplot(data=df, x= 'alpha', y='CL', markers='o')
plt.show()
print(df)

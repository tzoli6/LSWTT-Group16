import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

sns.set_theme()

df = pd.read_csv(".\CFD\\3D\\3D_windtunnel_model_T1-49_0_m_s-VLM1.csv", skipinitialspace=True)

sns.relplot(data=df, x='CD', y='CL', markers='o')
plt.show()
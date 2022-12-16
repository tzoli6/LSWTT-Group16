import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

sns.set_theme()



df = pd.read_csv(".\CFD\\3D\\3Dwindtunnelmodel_a=8.00_v=49.00ms_induced_angle.csv", skipinitialspace=True, delimiter=';')

sns.relplot(data=df, x='y-span', y='Ai', markers='o')
plt.show()
print(df)

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme()

df = pd.read_csv('G:\My Drive\TU Delft\LowSpeedWindTunnelTest\LSWTT-Group16\Measurement_Data\D\corr_test.csv')
print(df)
sns.relplot(data=df, x='Alpha' ,y='CL',markers='O')
#plt.plot(df['Alpha-pr'],df['Cl-press'])
plt.show()
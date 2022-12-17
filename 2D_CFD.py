import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import pandas as pd

#sns.set_theme()
#sns.set_style('darkgrid')

df = pd.read_csv('.\\CFD\\2D\data.csv', delimiter=',', skipinitialspace=True)
df.drop([0], inplace=True)

#f = so.Plot(data = df, x='alpha', y='CL')
#f.add(so.Dot())
plt.plot(df.loc[:,'alpha'], df.loc[:,'CD'])
plt.show()
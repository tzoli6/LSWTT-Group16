import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme()

df = pd.read_csv('.\Measurement_Data\\twoD\main.csv',skipinitialspace=True)
df = pd.DataFrame(df)

print(df)

#datapoints until stall (-3.0 - 16.5):
df_normal = df.drop(np.arange(26,43))

#datapoints from stall until max AoA (17.0 - 19.5):
df_stall = df.drop(np.arange(0,27))
df_stall = df_stall.drop(np.arange(33,43))

#datapoints in hysteresis (19.0H - 13.0H):
df_hyst = df.drop(np.arange(0,33))

#datapoints in normal + stall (-3.0 - 19.5):
df_ns = df.drop(np.arange(33,43))

#sns.scatterplot(data=df_normal, x='alpha' ,y='CL')
#sns.relplot(data=df_normal, x='alpha' ,y='CL',kind='line')

def plotter(x_axis, y_axis, dataset):
    labels = {
        "CL":dict(label = r'$C_{l}$', unit = '-', lower = 0, upper = 1.25, step = 0.25, name = 'Cl'),
        "alpha":dict(label = '\u03B1', unit = '\u00B0', lower = -4, upper = 20, step = 2.5, name = 'alpha'),
        "Cm":dict(label = r'$C_{m}$', unit = '-', lower = -0.1, upper = 0.04, step = 0.02, name = 'Cm'),
        "CD":dict(label = r'$C_{d}$', unit = '-', lower = 0, upper = 0.28, step = 0.04, name = 'Cd')
    }
    x_label = labels[x_axis]["label"]
    x_unit = labels[x_axis]["unit"]
    x_lower = labels[x_axis]["lower"]
    x_upper = labels[x_axis]["upper"]
    x_step = labels[x_axis]["step"]
    x_name = labels[x_axis]["name"]

    y_label = labels[y_axis]["label"]
    y_unit = labels[y_axis]["unit"]
    y_lower = labels[y_axis]["lower"]
    y_upper = labels[y_axis]["upper"]
    y_step = labels[y_axis]["step"]
    y_name = labels[y_axis]["name"]

    if dataset == 1:
        df_1 = df[[x_axis, y_axis]]
    if dataset == 2:
        df_1 = df_stall[[x_axis, y_axis]]
    if dataset == 3:
        df_1 = df_hyst[[x_axis, y_axis]]
    if dataset == 4:
        df_1 = df_ns[[x_axis, y_axis]]
    plt.plot(df_1[x_axis], df_1[y_axis], color='grey',linewidth='1')
    f=sns.scatterplot(data=df_1, x=df_1[x_axis], y=df_1[y_axis], marker ='X', color='seagreen')
    f.set(xlabel= x_label + ' [' + x_unit + ']', ylabel= y_label + ' [' + y_unit + ']')
    plt.yticks(np.arange(y_lower,y_upper,y_step))
    plt.xticks(np.arange(x_lower,x_upper,x_step))
    plt.savefig('.\Measurement_Data\\twoD\\Polars\\' + x_name + y_name + str(dataset) + '.svg',dpi=1200)
    plt.clf()
    #plt.show()

plotter('alpha', 'Cm', 1)
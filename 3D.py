import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme()

df = pd.read_csv('.\Measurement_Data\\3D\corr_test.csv',skipinitialspace=True)
df = pd.DataFrame(df)

#There were some datapoints which are redundant. These are the following:
#Run-nr
#29
#32
#33
#34

labels=[29-1,32-1,33-1,34-1]
df = df.drop(labels)
df = df.reset_index(drop=True)
print(df)

#datapoints until stall (-3.0 - 16.5):
df_normal = df.drop(np.arange(29,47))

#datapoints from stall until max AoA (17.0 - 19.5):
df_stall = df.drop(np.arange(0,30))
df_stall = df_stall.drop(np.arange(34,47))

#datapoints in hysteresis (19.0H - 13.0H):
df_hyst = df.drop(np.arange(0,34))

#datapoints in normal + stall (-3.0 - 19.5):
df_ns = df.drop(np.arange(34,47))

def plotter(x_axis, y_axis, dataset):
    labels = {
        "CL":dict(label = r'$C_{L}$', unit = '-', lower = 0, upper = 1.25, step = 0.25, name = 'CL'),
        "Alpha":dict(label = '\u03B1', unit = '\u00B0', lower = -4, upper = 20, step = 2.5, name = 'Alpha'),
        "Cm_pitch":dict(label = r'$C_{m}$', unit = '-', lower = -0.1, upper = 0.04, step = 0.02, name = 'CM'),
        "CD":dict(label = r'$C_{D}$', unit = '-', lower = 0, upper = 0.28, step = 0.04, name = 'CD')
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
    plt.savefig('.\Measurement_Data\\3D\\Polars\\' + x_name + y_name + str(dataset) + '.svg',dpi=1200)
    plt.clf()
    #plt.show()

plotter('CD', 'CL', 4)

# Alpha: \u03B1
# Degree: \u00B0

# ###CL-Alpha###
# df_1 = df[['Alpha','CL']]
# print(df_1)
# #plot all
# plt.plot(df_1['Alpha'], df_1['CL'], color='grey',linewidth='1')
# f=sns.scatterplot(data=df_1, x=df_1['Alpha'], y=df_1['CL'], marker ='X', color='seagreen')
# f.set(xlabel='Alpha [degrees]', ylabel='CL [-]')
# plt.show()
# #plot stall
# df_1_stall = df_stall[['Alpha','CL']]
# plt.plot(df_1_stall['Alpha'], df_1_stall['CL'], color='grey',linewidth='1')
# f=sns.scatterplot(data=df_1_stall, x=df_1['Alpha'], y=df_1_stall['CL'], marker ='X', color='seagreen')
# f.set(xlabel='Alpha [degrees]', ylabel='CL [-]')
# plt.show()
# #plot hysteresis
# df_1_hyst = df_hyst[['Alpha','CL']]
# plt.plot(df_1_hyst['Alpha'], df_1_hyst['CL'], color='grey',linewidth='1')
# f=sns.scatterplot(data=df_1_hyst, x=df_1['Alpha'], y=df_1_hyst['CL'], marker ='X', color='seagreen')
# f.set(xlabel='Alpha [degrees]', ylabel='CL [-]')
# plt.show()
#
# ###CL-CD###
#
# df_2 = df[['CD','CL']]
# #print(df_2)
# #plot
# plt.plot(df_2['CD'], df_2['CL'], color='grey',linewidth='1')
# f=sns.scatterplot(data=df_2, x=df_2['CD'], y=df_2['CL'], marker ='X', color='seagreen')
# f.set(xlabel='CD [-]', ylabel='CL [-]')
# plt.show()
#
#
# ###CD-Alpha###
#
# df_3 = df[['Alpha','CD']]
# #print(df_3)
# #plot
# plt.plot(df_3['Alpha'], df_3['CD'], color='grey',linewidth='1')
# f=sns.scatterplot(data=df_3, x=df_3['Alpha'], y=df_3['CD'], marker ='X', color='seagreen')
# f.set(xlabel='Alpha [degrees]', ylabel='CD [-]')
# plt.show()
#
# ###CM-Alpha###
#
# df_4 = df[['Alpha','Cm_pitch']]
# #print(df_4)
# #plot
# plt.plot(df_4['Alpha'], df_4['Cm_pitch'], color='grey',linewidth='1')
# f=sns.scatterplot(data=df_4, x=df_4['Alpha'], y=df_4['Cm_pitch'], marker ='X', color='seagreen')
# f.set(xlabel='Alpha [degrees]', ylabel='Cm [-]')
# plt.show()
#
# #sns.relplot(data=df, x='Alpha' ,y='CL',markers='O')
# #plt.plot(df['Alpha-pr'],df['Cl-press'])
# #plt.show()
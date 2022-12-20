import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
#os.chdir('..')
for i in range(-3,0):
    i = str(i)
    files = glob.glob('.\Measurement_Data\\3D_heat\\'+i+' deg\Record*')
    for j in range(0, len(files)):
        df = pd.read_csv(files[j])
        df = df.iloc[: , :-1]
        ma = df.values.max()
        mi = df.values.min()
        df = df.apply(lambda x: (x-mi)/(ma-mi))
        df = df.apply(lambda x: np.tanh(5*(x-0.73)))
        a = sns.set_palette(sns.color_palette(['#000000','#ffffff'], as_cmap=True))
        a = sns.color_palette('Greys_r', 1000)
        ax = sns.heatmap(df,cmap=a, yticklabels=False, xticklabels=False)
        print(df.iloc[225,220])
        plt.show()



#df = pd.read_csv('.\Measurement_Data\\3D_heat\\-1 deg\Record_2022-11-29_10-32-11.csv',delimiter=';', skipinitialspace=True)
#df = df.iloc[: , :-1]
#ma = df.values.max()
#mi = df.values.min()
#df = df.apply(lambda x: (x-mi)/(ma-mi))
#df = df.apply(lambda x: np.tanh(5*(x-0.73)))
#a = sns.set_palette(sns.color_palette(['#000000','#ffffff'], as_cmap=True))
#a = sns.color_palette('Greys_r', 1000)
#ax = sns.heatmap(df,cmap=a, yticklabels=False, xticklabels=False)
#print(df.iloc[225,220])
#plt.show()
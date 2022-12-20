import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
#os.chdir('..')

dflist = []

for i in np.arange(16,20):
    i = str(i)
    files = glob.glob('.\Measurement_Data\\2D_heat\\'+i+' deg 2.0\Record*')

    #puts ALL images into a list from ONE folder
    for j in range(0, len(files)):
        df = pd.read_csv(files[j], delimiter=';', skipinitialspace=True)
        df = df.iloc[: , :-1]
        dflist.append(df)
    
    df_avg = np.zeros(df.shape)
    
    for k in range(0,len(dflist)):
        addition = dflist[k]
        addition = addition.to_numpy()
        df_avg = df_avg + addition
    
    df_avg = pd.DataFrame(df_avg)
    df_avg = df_avg.apply(lambda x: x/len(files))
    
    ma = df_avg.values.max()
    mi = df_avg.values.min()
    df_avg = df_avg.apply(lambda x: (x-mi)/(ma-mi))
    df_avg = df_avg.apply(lambda x: np.tanh(4*(x-0.73)))
    a = sns.set_palette(sns.color_palette(['#000000','#ffffff'], as_cmap=True))
    a = sns.color_palette('Greys_r', 1000)
    label = str(i) + ' Deg_H'
    ax = sns.heatmap(df_avg, cmap=a, yticklabels=False, xticklabels=False, cbar = False)
    plt.text(5,20,label, color='#FFFFFF')
    plt.savefig('.\Measurement_Data\\2D_heat\Heat_figures\\'+label+'.png',dpi=250)
    plt.clf()
    dflist=[]
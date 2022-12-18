import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import pandas as pd
import numpy as np



sns.set_theme()
sns.set_style('darkgrid', {'axes.facecolor':'.9'})

df = pd.read_csv(".\CFD\\3D\\3D_windtunnel_model_T1-49_0_m_s-VLM1(2).csv", skipinitialspace=True)


sns.set(rc={'figure.figsize':(6,16)})
def save_cfd_plots():
    #CL-alpha
    f=sns.relplot(df, x='alpha', y='CL', kind='line', color='grey', linewidth='1', aspect=16/11)
    sns.scatterplot(df, x='alpha', y='CL', markers='x', color='seagreen')
    f.set(xlabel='\u03B1 [\u00B0]', ylabel='CL [-]')
    plt.yticks(np.arange(-1,1.5,0.25))
    plt.xticks(np.arange(-3,20, 1))
    plt.savefig('.\CFD\\3D\graphs\VLM_CFD_CL-alpha.svg',dpi=1200)
    plt.clf()
    #CD-alpha
    f=sns.relplot(df, x='alpha', y='CD', kind='line', color='grey', linewidth='1', aspect=16/11)
    sns.scatterplot(df, x='alpha', y='CD', markers='x', color='seagreen')
    f.set(xlabel='\u03B1 [\u00B0]', ylabel='CD [-]')
    plt.yticks(np.arange(-0.05,0.35,0.02))
    plt.xticks(np.arange(-3,20, 1))
    plt.savefig('.\CFD\\3D\graphs\VLM_CFD_CD-alpha.svg',dpi=1200)
    plt.clf()
    #CD-CL
    plt.plot(df.loc[:,'CD'], df.loc[:,'CL'], color='grey',linewidth='1')
    f= sns.scatterplot(df, x='CD', y='CL', markers='x', color='seagreen')
    f.set(xlabel='CD [-]', ylabel='CL [-]')
    plt.yticks(np.arange(-1,1.5,0.2))
    plt.xticks(np.arange(-0.05,0.15,0.02))
    plt.savefig('.\CFD\\3D\graphs\VLM_CFD_CL-CD.svg',dpi=1200)
    plt.clf()
    #Cm-alpha
    f=sns.relplot(df, x='alpha', y='Cm', kind='line', color='grey', linewidth='1', aspect=16/11)
    sns.scatterplot(df, x='alpha', y='Cm', markers='x', color='seagreen')
    f.set(xlabel='\u03B1 [\u00B0]', ylabel='Cm [-]')
    plt.yticks(np.arange(-0.3,0.06,0.01))
    plt.xticks(np.arange(-3,22.5,2.5))
    plt.savefig('.\CFD\\3D\graphs\VLM_CFD_Cm-alpha.svg',dpi=1200)
    plt.clf()
    #CDi-alpha
    f = sns.relplot(df, x='alpha', y='CDi', kind='line', color='grey', linewidth='1', aspect=16/11)
    sns.scatterplot(df, x='alpha', y='CDi', markers='x', color='seagreen')
    f.set(xlabel='\u03B1 [\u00B0]', ylabel='CDi [-]')
    plt.yticks(np.arange(-0.05, 0.15, 0.02))
    plt.xticks(np.arange(-3, 20, 1))
    plt.savefig('.\CFD\\3D\graphs\VLM_CFD_CDi-alpha.svg', dpi=1200)
    plt.clf()


def main():
    save_cfd_plots()

if __name__=="__main__":
    main()


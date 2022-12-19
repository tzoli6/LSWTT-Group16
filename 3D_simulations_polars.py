import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import pandas as pd
import numpy as np



sns.set_theme()
sns.set_style('darkgrid', {'axes.facecolor':'.9'})

df_1 = pd.read_csv('.\\CFD\\3D\\3D_windtunnel_model_T1-49_0_m_s-LLT_V2(4).csv', skipinitialspace=True)
df_2 = pd.read_csv(".\CFD\\3D\\3D_windtunnel_model_T1-49_0_m_s-VLM1(2).csv", skipinitialspace=True)
df_3 = pd.read_csv(".\CFD\\3D\\3D_windtunnel_model_T1-49_0_m_s-Panel(2).csv", skipinitialspace=True)

df_con= pd.concat([df_1.assign(dataset='LLT'), df_2.assign(dataset='VLM'), df_3.assign(dataset='Panel')])
print(df_con)
sns.set(rc={'figure.figsize':(6,16)})

"""
def save_cfd_plots():
    #CL-alpha
    f=sns.relplot(df_2, x='alpha', y='CL', kind='line', color='grey', linewidth='1', aspect=16/11)
    sns.scatterplot(df_2, x='alpha', y='CL', markers='x', color='seagreen')
    f.set(xlabel='\u03B1 [\u00B0]', ylabel='CL [-]')
    plt.yticks(np.arange(-1,1.5,0.25))
    plt.xticks(np.arange(-3,20, 1))
    plt.savefig('.\CFD\\3D\graphs\VLM_CFD_CL-alpha.svg',dpi=1200)
    plt.clf()
    #CD-alpha
    f=sns.relplot(df_2, x='alpha', y='CD', kind='line', color='grey', linewidth='1', aspect=16/11)
    sns.scatterplot(df_2, x='alpha', y='CD', markers='x', color='seagreen')
    f.set(xlabel='\u03B1 [\u00B0]', ylabel='CD [-]')
    plt.yticks(np.arange(-0.05,0.35,0.02))
    plt.xticks(np.arange(-3,20, 1))
    plt.savefig('.\CFD\\3D\graphs\VLM_CFD_CD-alpha.svg',dpi=1200)
    plt.clf()
    #CD-CL
    plt.plot(df_2.loc[:,'CD'], df_2.loc[:,'CL'], color='grey',linewidth='1')
    f= sns.scatterplot(df_2, x='CD', y='CL', markers='x', color='seagreen')
    f.set(xlabel='CD [-]', ylabel='CL [-]')
    plt.yticks(np.arange(-1,1.5,0.2))
    plt.xticks(np.arange(-0.05,0.15,0.02))
    plt.savefig('.\CFD\\3D\graphs\VLM_CFD_CL-CD.svg',dpi=1200)
    plt.clf()
    #Cm-alpha
    f=sns.relplot(df_2, x='alpha', y='Cm', kind='line', color='grey', linewidth='1', aspect=16/11)
    sns.scatterplot(df_2, x='alpha', y='Cm', markers='x', color='seagreen')
    f.set(xlabel='\u03B1 [\u00B0]', ylabel='Cm [-]')
    plt.yticks(np.arange(-0.3,0.06,0.01))
    plt.xticks(np.arange(-3,22.5,2.5))
    plt.savefig('.\CFD\\3D\graphs\VLM_CFD_Cm-alpha.svg',dpi=1200)
    plt.clf()
    #CDi-alpha
    f = sns.relplot(df_2, x='alpha', y='CDi', kind='line', color='grey', linewidth='1', aspect=16/11)
    sns.scatterplot(df_2, x='alpha', y='CDi', markers='x', color='seagreen')
    f.set(xlabel='\u03B1 [\u00B0]', ylabel='CDi [-]')
    plt.yticks(np.arange(-0.05, 0.15, 0.02))
    plt.xticks(np.arange(-3, 20, 1))
    plt.savefig('.\CFD\\3D\graphs\VLM_CFD_CDi-alpha.svg', dpi=1200)
    plt.clf()
"""

def save_comparison_plots():
    sns.set_palette('Paired')

    # CL-alpha
    g = sns.relplot(df_con, x='alpha', y='CL', kind='scatter', style='dataset', hue='dataset', aspect=16 / 11, legend=False)
    sns.scatterplot(df_con, x='alpha', y='CL', style='dataset', hue='dataset')
    g.axes.flat[0].get_legend().set_title('')
    plt.plot(df_1.loc[:, 'alpha'], df_1.loc[:, 'CL'], color='grey', linewidth='1')
    plt.plot(df_2.loc[:, 'alpha'], df_2.loc[:, 'CL'], color='grey', linewidth='1')
    plt.plot(df_3.loc[:, 'alpha'], df_3.loc[:, 'CL'], color='grey', linewidth='1')
    g.set(xlabel='\u03B1 [\u00B0]', ylabel='Cl [-]')
    plt.yticks(np.arange(-0.5, 1.5, 0.25))
    plt.xticks(np.arange(-5, 22.5, 2.5))
    print('1')
    plt.savefig('.\CFD\\3D\graphs\simulations_CL-alpha.svg', dpi=1200)
    print('2')
    plt.clf()

    # CD-alpha
    g = sns.relplot(df_con, x='alpha', y='CD', kind='scatter', style='dataset', hue='dataset', aspect=16 / 11, legend=False)  # linewidth='1',color='grey'
    sns.scatterplot(df_con, x='alpha', y='CD', style='dataset', hue='dataset')
    g.axes.flat[0].get_legend().set_title('')
    plt.plot(df_1.loc[:, 'alpha'], df_1.loc[:, 'CD'], color='grey', linewidth='1')
    plt.plot(df_2.loc[:, 'alpha'], df_2.loc[:, 'CD'], color='grey', linewidth='1')
    plt.plot(df_3.loc[:, 'alpha'], df_3.loc[:, 'CD'], color='grey', linewidth='1')
    g.set(xlabel='\u03B1 [\u00B0]', ylabel='CD [-]')
    plt.yticks(np.arange(0, 0.15, 0.05))
    plt.xticks(np.arange(-5, 20, 2.5))
    print(3)
    plt.savefig('.\CFD\\3D\graphs\simulations_CD-alpha.svg', dpi=1200)
    plt.clf()

    # CD-CL
    g = sns.relplot(df_con, x='CD', y='CL', kind='scatter', style='dataset', hue='dataset', aspect=16 / 11, legend=False)
    sns.scatterplot(df_con, x='CD', y='CL', style='dataset', hue='dataset')
    g.axes.flat[0].get_legend().set_title('')
    plt.plot(df_1.loc[:, 'CD'], df_1.loc[:, 'CL'], color='grey', linewidth='1')
    plt.plot(df_2.loc[:, 'CD'], df_2.loc[:, 'CL'], color='grey', linewidth='1')
    plt.plot(df_3.loc[:, 'CD'], df_3.loc[:, 'CL'], color='grey', linewidth='1')
    g.set(xlabel='CD [-]', ylabel='CL [-]')
    plt.yticks(np.arange(-0.5, 1.5, 0.25))
    plt.xticks(np.arange(-0.05, 0.15, 0.05))
    plt.savefig('.\CFD\\3D\graphs\simulations_CD-CL.svg', dpi=1200)
    plt.clf()

    # Cm-alpha
    g = sns.relplot(df_con, x='alpha', y='Cm', kind='scatter', style='dataset', hue='dataset', aspect=16 / 11, legend=False)
    sns.scatterplot(df_con, x='alpha', y='Cm', style='dataset', hue='dataset')
    g.axes.flat[0].get_legend().set_title('')
    plt.plot(df_1.loc[:, 'alpha'], df_1.loc[:, 'Cm'], color='grey', linewidth='1')
    plt.plot(df_2.loc[:, 'alpha'], df_2.loc[:, 'Cm'], color='grey', linewidth='1')
    plt.plot(df_3.loc[:, 'alpha'], df_3.loc[:, 'Cm'], color='grey', linewidth='1')
    g.set(xlabel='\u03B1 [\u00B0]', ylabel='Cm [-]')
    plt.yticks(np.arange(-0.3, 0.06, 0.05))
    plt.xticks(np.arange(-5, 22.5, 2.5))
    plt.savefig('.\CFD\\3D\graphs\simulations_Cm-alpha.svg', dpi=1200)
    plt.clf()

    # CDi-alpha
    # CD-alpha
    g = sns.relplot(df_con, x='alpha', y='CDi', kind='scatter', style='dataset', hue='dataset', aspect=16 / 11, legend=False)  # linewidth='1',color='grey'
    sns.scatterplot(df_con, x='alpha', y='CDi', style='dataset', hue='dataset')
    g.axes.flat[0].get_legend().set_title('')
    plt.plot(df_1.loc[:, 'alpha'], df_1.loc[:, 'CDi'], color='grey', linewidth='1')
    plt.plot(df_2.loc[:, 'alpha'], df_2.loc[:, 'CDi'], color='grey', linewidth='1')
    plt.plot(df_3.loc[:, 'alpha'], df_3.loc[:, 'CDi'], color='grey', linewidth='1')
    g.set(xlabel='\u03B1 [\u00B0]', ylabel='CDi [-]')
    plt.yticks(np.arange(0, 0.15, 0.05))
    plt.xticks(np.arange(-5, 20, 2.5))
    print(3)
    plt.savefig('.\CFD\\3D\graphs\simulations_CDi-alpha.svg', dpi=1200)
    plt.clf()


def main():
    #save_cfd_plots()
    save_comparison_plots()

if __name__=="__main__":
    main()


import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import pandas as pd
import numpy as np



sns.set_theme()
sns.set_style('darkgrid', {'axes.facecolor':'.9'})

df = pd.read_csv('.\\CFD\\2D\data.csv', delimiter=',', skipinitialspace=True)
df.drop([0], inplace=True)
df = df.astype(float)

df_data = pd.read_csv('.\Measurement_Data\\twoD\main.csv',skipinitialspace=True)

df_con= pd.concat([df.assign(dataset='CFD'), df_data.assign(dataset='Measurement')])
#print(df)
sns.set(rc={'figure.figsize':(6,16)})

#sns.relplot(df, x='alpha', y='CL', style='dataset', kind='line', color='grey', legend=False)
#sns.scatterplot(df, x='alpha', y='CL', style='dataset')
#sns.relplot(df_data, x='Alpha', y='Cl', kind='line', color='grey')
#sns.scatterplot(df_data, x= 'Alpha', y='Cl')
def save_cfd_plots():
    #Cl-alpha
    f=sns.relplot(df, x='alpha', y='CL', kind='line', color='grey', linewidth='1', aspect=16/11)
    sns.scatterplot(df, x='alpha', y='CL', markers='x', color='seagreen')
    f.set(xlabel='\u03B1 [\u00B0]', ylabel='Cl [-]')
    plt.yticks(np.arange(-1,1.5,0.25))
    plt.xticks(np.arange(-10,20,2.5))
    #plt.savefig('.\CFD\\2D\Graphs\CFD_Cl-alpha.svg',dpi=1200)
    plt.clf()
    #cd-alpha
    f=sns.relplot(df, x='alpha', y='CD', kind='line', color='grey', linewidth='1', aspect=16/11)
    sns.scatterplot(df, x='alpha', y='CD', markers='x', color='seagreen')
    f.set(xlabel='\u03B1 [\u00B0]', ylabel='Cd [-]')
    plt.yticks(np.arange(-0.05,0.15,0.02))
    plt.xticks(np.arange(-10,20,2.5))
    #plt.savefig('.\CFD\\2D\Graphs\CFD_Cd-alpha.svg',dpi=1200)
    plt.clf()
    #cd-cl
    plt.plot(df.loc[:,'CD'], df.loc[:,'CL'], color='grey',linewidth='1')
    f= sns.scatterplot(df, x='CD', y='CL', markers='x', color='seagreen')
    f.set(xlabel='Cd [-]', ylabel='Cl [-]')
    plt.yticks(np.arange(-1,1.5,0.2))
    plt.xticks(np.arange(-0.05,0.15,0.02))
    #plt.savefig('.\CFD\\2D\Graphs\CFD_Cl-Cd.svg',dpi=1200)
    plt.clf()
    #cm-alpha
    f=sns.relplot(df, x='alpha', y='Cm', kind='line', color='grey', linewidth='1', aspect=16/11)
    sns.scatterplot(df, x='alpha', y='Cm', markers='x', color='seagreen')
    f.set(xlabel='\u03B1 [\u00B0]', ylabel='Cm [-]')
    plt.yticks(np.arange(-0.03,0.06,0.01))
    plt.xticks(np.arange(-10,22.5,2.5))
    plt.savefig('.\CFD\\2D\Graphs\CFD_Cm-alpha.svg',dpi=1200)
    plt.clf()

def save_comperison_plots():
    sns.set_palette('Paired')
    #cl-alpha
    g=sns.relplot(df, x='alpha', y='CL', kind='line', linewidth='1',color='grey', aspect=16/11)
    sns.scatterplot(df_con, x='alpha', y='CL', style='dataset', hue='dataset')
    g.axes.flat[0].get_legend().set_title('')
    plt.plot(df_data.loc[:,'alpha'], df_data.loc[:,'CL'], color='grey',linewidth='1')
    g.set(xlabel='\u03B1 [\u00B0]', ylabel='Cl [-]')
    plt.yticks(np.arange(-1,1.5,0.25))
    plt.xticks(np.arange(-10,22.5,2.5))
    plt.savefig('.\CFD\\2D\Graphs\CFD_M_Cl-alpha.svg',dpi=1200)
    plt.clf()
    #Cd-alpha
    g=sns.relplot(df, x='alpha', y='CD', kind='line', linewidth='1',color='grey', aspect=16/11)
    sns.scatterplot(df_con, x='alpha', y='CD', style='dataset', hue='dataset')
    g.axes.flat[0].get_legend().set_title('')
    plt.plot(df_data.loc[:,'alpha'], df_data.loc[:,'CD'], color='grey',linewidth='1')
    g.set(xlabel='\u03B1 [\u00B0]', ylabel='Cd [-]')
    plt.yticks(np.arange(-0.05,0.35,0.05))
    plt.xticks(np.arange(-10,20,2.5))
    plt.savefig('.\CFD\\2D\Graphs\CFD_M_Cd-alpha.svg',dpi=1200)
    plt.clf()

    #Cd-Cl
    g=sns.relplot(df_con, x='CD', y='CL',kind='scatter', style='dataset', hue='dataset', aspect=16/11,legend=False)
    sns.scatterplot(df_con, x='CD', y='CL', style='dataset', hue='dataset')
    g.axes.flat[0].get_legend().set_title('')
    plt.plot(df_data.loc[:,'CD'], df_data.loc[:,'CL'], color='grey',linewidth='1')
    plt.plot(df.loc[:,'CD'], df.loc[:,'CL'], color='grey',linewidth='1')
    g.set(xlabel='Cl [-]', ylabel='Cd [-]')
    plt.yticks(np.arange(-1,1.5,0.25))
    plt.xticks(np.arange(-0.05,0.35,0.05))
    plt.savefig('.\CFD\\2D\Graphs\CFD_M_Cd-Cl.svg',dpi=1200)
    plt.clf()
#Cm-alpha
    g=sns.relplot(df, x='alpha', y='Cm', kind='line', linewidth='1',color='grey', aspect=16/11)
    sns.scatterplot(df_con, x='alpha', y='Cm', style='dataset', hue='dataset')
    g.axes.flat[0].get_legend().set_title('')
    plt.plot(df_data.loc[:,'alpha'], df_data.loc[:,'Cm'], color='grey',linewidth='1')
    g.set(xlabel='\u03B1 [\u00B0]', ylabel='Cm [-]')
    plt.yticks(np.arange(-0.1,0.06,0.01))
    plt.xticks(np.arange(-10,22.5,2.5))
    plt.savefig('.\CFD\\2D\Graphs\CFD_M_Cm-alpha.svg',dpi=1200)
    plt.clf()

def save_CFD_Cp_plots():
    df = pd.read_csv('.\\CFD\\2D\\Cp\\17.5.txt', delimiter=' ', skipinitialspace=True)
    print(df)
    df = df.astype(float)
    df = df[df.index % 3 ==0]
    f=sns.relplot(df, x='x', y='Cpv', kind='scatter', aspect=16/11)
    f=sns.scatterplot(df, x='x', y='Cpv', markers='x', color='seagreen')
    plt.plot(df.loc[:,'x'], df.loc[:,'Cpv'], color = 'grey', linewidth=1)
    f.set(xlabel='x/c [-]', ylabel='Cp [-]')
    f.invert_yaxis()
    plt.savefig('.\CFD\\2D\Graphs\Cp_17.5.svg',dpi=1200)
    plt.clf()

def main():
    #save_cfd_plots()
    #save_comperison_plots()
    save_CFD_Cp_plots()



'''
f = so.Plot(data = df, x='alpha', y='CL')
f.add(so.Line(color='grey'))
f.add(so.Dot(marker='o', color='seagreen'))
f.show()
(so.Plot(data=df, x=['alpha','alpha'], y=['Cm','CL'])
 .add(so.Line())
 .show()
)
#plt.plot(df.loc[:,'alpha'], df.loc[:,'CD'])
'''
if __name__=="__main__":
    main()
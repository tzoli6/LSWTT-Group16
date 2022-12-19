import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np



sns.set_theme()
sns.set_style('darkgrid', {'axes.facecolor':'.9'})

df = pd.read_csv('.\\CFD\\3D\\3D_windtunnel_model_T1-49_0_m_s-LLT_V2(4).csv', skipinitialspace=True)


df_data = pd.read_csv('.\Measurement_Data\\3D\corr_test_V2.csv',skipinitialspace=True)

df_data = pd.DataFrame(df_data)

#There were some datapoints which are redundant. These are the following:
#Run-nr
#29
#32
#33
#34

labels=[29-1,32-1,33-1,34-1]
df_data = df_data.drop(labels)
df_data = df_data.reset_index(drop=True)
print(df_data)

#datapoints until stall (-3.0 - 16.5):
df_normal = df_data.drop(np.arange(29,47))

#datapoints from stall until max AoA (17.0 - 19.5):
df_stall = df_data.drop(np.arange(0,30))
df_stall = df_stall.drop(np.arange(34,47))

#datapoints in hysteresis (19.0H - 13.0H):
df_hyst = df_data.drop(np.arange(0,34))

'''
###CL-Alpha###
df_1 = df_data[['Alpha','CL']]
print(df_1)
#plot all
plt.plot(df_1['Alpha'], df_1['CL'], color='grey',linewidth='1')
f=sns.scatterplot(data=df_1, x=df_1['Alpha'], y=df_1['CL'], marker ='X', color='seagreen')
f.set(xlabel='Alpha [degrees]', ylabel='CL [-]')
plt.show()
#plot stall
df_1_stall = df_stall[['Alpha','CL']]
plt.plot(df_1_stall['Alpha'], df_1_stall['CL'], color='grey',linewidth='1')
f=sns.scatterplot(data=df_1_stall, x=df_1['Alpha'], y=df_1_stall['CL'], marker ='X', color='seagreen')
f.set(xlabel='Alpha [degrees]', ylabel='CL [-]')
plt.show()
#plot hysteresis
df_1_hyst = df_hyst[['Alpha','CL']]
plt.plot(df_1_hyst['Alpha'], df_1_hyst['CL'], color='grey',linewidth='1')
f=sns.scatterplot(data=df_1_hyst, x=df_1['Alpha'], y=df_1_hyst['CL'], marker ='X', color='seagreen')
f.set(xlabel='Alpha [degrees]', ylabel='CL [-]')
plt.show()

###CL-CD###

df_2 = df_data[['CD','CL']]
#print(df_2)
#plot
plt.plot(df_2['CD'], df_2['CL'], color='grey',linewidth='1')
f=sns.scatterplot(data=df_2, x=df_2['CD'], y=df_2['CL'], marker ='X', color='seagreen')
f.set(xlabel='CD [-]', ylabel='CL [-]')
plt.show()


###CD-Alpha###

df_3 = df_data[['Alpha','CD']]
#print(df_3)
#plot
plt.plot(df_3['Alpha'], df_3['CD'], color='grey',linewidth='1')
f=sns.scatterplot(data=df_3, x=df_3['Alpha'], y=df_3['CD'], marker ='X', color='seagreen')
f.set(xlabel='Alpha [degrees]', ylabel='CD [-]')
plt.show()

###CM-Alpha###

df_4 = df_data[['Alpha','Cm_pitch']]
#print(df_4)
#plot
plt.plot(df_4['Alpha'], df_4['Cm_pitch'], color='grey',linewidth='1')
f=sns.scatterplot(data=df_4, x=df_4['Alpha'], y=df_4['Cm_pitch'], marker ='X', color='seagreen')
f.set(xlabel='Alpha [degrees]', ylabel='Cm [-]')
plt.show()
'''
#sns.relplot(data=df, x='Alpha' ,y='CL',markers='O')
#plt.plot(df['Alpha-pr'],df['Cl-press'])
#plt.show()
df_con= pd.concat([df.assign(dataset='CFD'), df_data.assign(dataset='Measurement')])
print(df)
sns.set(rc={'figure.figsize':(6,16)})


#sns.relplot(df, x='alpha', y='CL', style='dataset', kind='line', color='grey', legend=False)
#sns.scatterplot(df, x='alpha', y='CL', style='dataset')
#sns.relplot(df_data, x='Alpha', y='Cl', kind='line', color='grey')
#sns.scatterplot(df_data, x= 'Alpha', y='Cl')
def save_cfd_plots():
    #CL-alpha
    f=sns.relplot(df, x='alpha', y='CL', kind='line', color='grey', linewidth='1', aspect=16/11)
    sns.scatterplot(df, x='alpha', y='CL', markers='x', color='seagreen')
    f.set(xlabel='\u03B1 [\u00B0]', ylabel='CL [-]')
    plt.yticks(np.arange(-1,1.5,0.25))
    plt.xticks(np.arange(-10,20, 1))
    #plt.savefig('.\CFD\\3D\graphs\LLT_CFD_CL-alpha.svg',dpi=1200)
    plt.clf()
    #CD-alpha
    f=sns.relplot(df, x='alpha', y='CD', kind='line', color='grey', linewidth='1', aspect=16/11)
    sns.scatterplot(df, x='alpha', y='CD', markers='x', color='seagreen')
    f.set(xlabel='\u03B1 [\u00B0]', ylabel='CD [-]')
    plt.yticks(np.arange(-0.05,0.35,0.02))
    plt.xticks(np.arange(-10,20, 1))
    #plt.savefig('.\CFD\\3D\graphs\LLT_CFD_CD-alpha.svg',dpi=1200)
    plt.clf()
    #CD-CL
    plt.plot(df.loc[:,'CD'], df.loc[:,'CL'], color='grey',linewidth='1')
    f= sns.scatterplot(df, x='CD', y='CL', markers='x', color='seagreen')
    f.set(xlabel='CD [-]', ylabel='CL [-]')
    plt.yticks(np.arange(-1,1.5,0.2))
    plt.xticks(np.arange(-0.05,0.15,0.02))
    #plt.savefig('.\CFD\\3D\graphs\LLT_CFD_CL-CD.svg',dpi=1200)
    plt.clf()
    #Cm-alpha
    f=sns.relplot(df, x='alpha', y='Cm', kind='line', color='grey', linewidth='1', aspect=16/11)
    sns.scatterplot(df, x='alpha', y='Cm', markers='x', color='seagreen')
    f.set(xlabel='\u03B1 [\u00B0]', ylabel='Cm [-]')
    plt.yticks(np.arange(-0.3,0.06,0.01))
    plt.xticks(np.arange(-10,22.5,2.5))
    #plt.savefig('.\CFD\\3D\graphs\LLT_CFD_Cm-alpha.svg',dpi=1200)
    plt.clf()
    #CDi-alpha
    f = sns.relplot(df, x='alpha', y='CDi', kind='line', color='grey', linewidth='1', aspect=16/11)
    sns.scatterplot(df, x='alpha', y='CDi', markers='x', color='seagreen')
    f.set(xlabel='\u03B1 [\u00B0]', ylabel='CDi [-]')
    plt.yticks(np.arange(-0.05, 0.15, 0.02))
    plt.xticks(np.arange(-10, 20, 1))
    plt.savefig('.\CFD\\3D\graphs\LLT_CFD_CDi-alpha.svg', dpi=1200)
    plt.clf()

def save_comparison_plots():
    sns.set_palette('Paired')

    #CL-alpha
    g=sns.relplot(df_con, x='alpha', y='CL', kind='scatter', linewidth='1',color='grey', aspect=16/11, legend=False)
    sns.scatterplot(df_con, x='alpha', y='CL', style='dataset', hue='dataset')
    g.axes.flat[0].get_legend().set_title('')
    plt.plot(df_data.loc[:,'alpha'], df_data.loc[:,'CL'], color='grey',linewidth='1')
    g.set(xlabel='\u03B1 [\u00B0]', ylabel='Cl [-]')
    plt.yticks(np.arange(-1,1.5,0.25))
    plt.xticks(np.arange(-10,22.5,2.5))
    print('1')
    plt.savefig('.\CFD\\3D\graphs\CFD-Experimental_CL-alpha.svg',dpi=1200)
    print('2')
    plt.clf()

    #CD-alpha
    g=sns.relplot(df_con, x='alpha', y='CD', kind='scatter', style='dataset', hue='dataset', aspect=16/11, legend=False) #linewidth='1',color='grey'
    sns.scatterplot(df_con, x='alpha', y='CD', style='dataset', hue='dataset')
    g.axes.flat[0].get_legend().set_title('')
    plt.plot(df_data.loc[:,'alpha'], df_data.loc[:,'CD'], color='grey',linewidth='1')
    plt.plot(df.loc[:,'alpha'], df.loc[:,'CD'], color='grey',linewidth='1')
    g.set(xlabel='\u03B1 [\u00B0]', ylabel='CD [-]')
    plt.yticks(np.arange(-0.05,0.35,0.05))
    plt.xticks(np.arange(-10,20,2.5))
    print(3)
    plt.savefig('.\CFD\\3D\graphs\CFD-Experimental_CD-alpha.svg',dpi=1200)
    plt.clf()

    #CD-CL
    g=sns.relplot(df_con, x='CD', y='CL',kind='scatter', style='dataset', hue='dataset', aspect=16/11,legend=False)
    sns.scatterplot(df_con, x='CD', y='CL', style='dataset', hue='dataset')
    g.axes.flat[0].get_legend().set_title('')
    plt.plot(df_data.loc[:,'CD'], df_data.loc[:,'CL'], color='grey',linewidth='1')
    plt.plot(df.loc[:,'CD'], df.loc[:,'CL'], color='grey',linewidth='1')
    g.set(xlabel='CL [-]', ylabel='CD [-]')
    plt.yticks(np.arange(-1,1.5,0.25))
    plt.xticks(np.arange(-0.05,0.35,0.05))
    plt.savefig('.\CFD\\3D\graphs\CFD-Experimental_CD-CL.svg',dpi=1200)
    plt.clf()
    
    #Cm-alpha
    g=sns.relplot(df_con, x='alpha', y='Cm', kind='line', linewidth='1',color='grey', aspect=16/11, legend=False)
    sns.scatterplot(df_con, x='alpha', y='Cm', style='dataset', hue='dataset')
    g.axes.flat[0].get_legend().set_title('')
    plt.plot(df_data.loc[:,'alpha'], df_data.loc[:,'Cm'], color='grey',linewidth='1')
    g.set(xlabel='\u03B1 [\u00B0]', ylabel='Cm [-]')
    plt.yticks(np.arange(-0.3,0.06,0.01))
    plt.xticks(np.arange(-10,22.5,2.5))
    plt.savefig('.\CFD\\3D\graphs\CFD-Experimental_Cm-alpha.svg',dpi=1200)
    plt.clf()

def main():
    #save_cfd_plots()
    save_comparison_plots()



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
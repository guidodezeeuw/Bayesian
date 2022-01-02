# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def find_columns(cpt):
    """
    Function to read gef file and find which columns contain what values
    """
    data = open(cpt,'r')
    
    for row in data:
        if row[0]=='#':
            if row[-11:-1] == 'diepte, 11':
                lengthcol = int(row[13])-1
            elif row[-18:-1] == 'conusweerstand, 2':
                conuscol = int(row[13])-1      
            elif row[-25:-1] == 'plaatselijke wrijving, 3':
                sleevecol = int(row[13])-1
            elif row[-20:-1] == 'waterspanning u2, 6':
                u2col = int(row[13])-1
        else:
            break
    columns = [lengthcol, conuscol, sleevecol, u2col]
    columns = np.sort(columns)
    return columns


def read_cpt(params): 
    """
    Function to read cpt data and store them in a matrix
    """
    
    cpt = params["cpt"]
    no_of_columns = 12 #if more columns are included, add those here
    #0 = length
    #1 = conus
    #2 = sleeve
    #3 = u2
    #4 = fr
    #5 = I_sbt
    #6 = volumetric weigth soil
    #7 = total vertical stress
    #8 = effective vertical stress assuming ps is at z=0
    #9 = normalized cone stress
    #10 = normalized friction ratio
    #11 = corrected cone resistance
    
    """ Error values """
    lengtherror = 999.999
    conuserror = 999.999
    sleeveerror = 9.999
    u2error = 99.999
    
    pa = 0.1        #atmospheric pressure
    yw = 10         #unit weight of water
    alpha = 0.8     #coefficient to correct for pore pressures

    """ read columns from gef file """
    columns = find_columns(cpt) #read the columns from gef file
    df = pd.read_csv(cpt, header=None, delimiter=';',comment='#', usecols=columns)
    matrix = np.zeros((len(df),no_of_columns)) #make matrix in which data is stored
    #0: depth
    matrix[:,0] = np.array(df.iloc[:,1])
    
    #1: conus resistance
    matrix[:,1] = np.array(df.iloc[:,0])
    
    #2: sleeve friction
    matrix[:,2] = np.array(df.iloc[:,2])
    
    #3: u2
    matrix[:,3] = (matrix[:,0] * 0.01) -0.5*0.01
    matrix[:,3] = matrix[:,3]*(matrix[:,3]>0)


    #check if first value has error and if so, give it the closest 'real' value
    if matrix[0,0] == lengtherror:
        matrix[0,0]=matrix[np.where(matrix[:,0]!=lengtherror)[0][0],0]
    if matrix[0,1] == conuserror:
        matrix[0,1]=matrix[np.where(matrix[:,1]!=conuserror)[0][0],1]
    if matrix[0,2] == sleeveerror:
        matrix[0,2]=matrix[np.where(matrix[:,2]!=sleeveerror)[0][0],2]
    if matrix[0,3] == u2error:
        matrix[0,3]=matrix[np.where(matrix[:,3]!=u2error)[0][0],3]

    #if errorvalue is found, change it to the value above
    for point in range(len(matrix)):
        if matrix[point,0] == lengtherror:
            matrix[point,0] = matrix[point-1,0]
        if matrix[point,1] == conuserror:
            matrix[point,1] = matrix[point-1,1]
        if matrix[point,2] == sleeveerror:
            matrix[point,2] = matrix[point-1,2]
        if matrix[point,3] == u2error:
            matrix[point,3] = matrix[point-1,3]

    """ calculate other columns """
    
    #11: corrected cone resistance
    matrix[:,11] =  matrix[:,1]
    
    #4: fr
    matrix[:,4]= (matrix[:,2]/matrix[:,11])*100
    
    #5: Isbt
    matrix[:,5] = ((3.47-np.log10(matrix[:,11]/pa))**2)
    
    #6: volumetric soil weight kN/m3 (Robertson, 1990)
    matrix[:,6] = (yw*(0.27*np.log10(matrix[:,4])+0.36*np.log10(matrix[:,11]/pa)+1.236))
    for ii in range(len(matrix)):
        if matrix[ii,6] == np.inf or matrix[ii,6] == -np.inf:
            matrix[ii,6] = matrix[ii-1,6]
    
    #7: total vertical stress
    matrix[0,7] = matrix[0,6]*matrix[0,0]/1000 
    for ii in range(len(matrix)-1):
        matrix[ii+1,7] = matrix[ii,7]+(((matrix[ii+1,0]-matrix[ii,0])*matrix[ii+1,6])/1000)
        if math.isnan(matrix[ii+1,7])==True:
            matrix[ii+1,7] = matrix[ii,7]
        
    #8: effective vertical stress
    matrix[:,8] = matrix[:,7]-matrix[:,3]

    #9: normalized cone resistance    
    matrix[:,9] = (matrix[:,11]-matrix[:,7])/matrix[:,8]
    
    #10: normalized friction ratio
    matrix[:,10] = matrix[:,2]/(matrix[:,11]-matrix[:,7])*100

    return matrix


def generalize(matrix, params):
    """
    Function to generalized the cpt matrix by a ratio based on the min_thickness. 
    e.g. if min_thickness = 0.02 (= the general spacing between 2 cpt measurements), no generalization will be done
    e.g. if min_thickness = 0.1, every 5 points (=0.1/0.02) an average value will be taken for nQt and nFr.
    in addition 9 new (empty) columns are added in which the probability of MC are be stored.
    """
    min_thickness = params["min thickness"]
    no_of_columns = 12
    #0 = depth generalized
    #1 = log of nQt generalized
    #2 = log of nFr generalized
    #3-11 = for probabilities; for now empty
    
    generalize_ratio = min_thickness/0.02
    length_generalized = math.floor(round(len(matrix)/generalize_ratio,1)) #new length of matrix
    matrix_generalized = np.zeros((length_generalized,no_of_columns)) #make matrix in which data is stored
    
    #0 = depth generalized
    matrix_generalized[:,0]= np.mean(matrix[0:int(length_generalized*generalize_ratio),0].reshape(-1,int(generalize_ratio)),axis=1)
    
    #1 = log of nQt generalized
    matrix_generalized[:,1]= np.log(np.mean(matrix[0:int(length_generalized*generalize_ratio),9].reshape(-1,int(generalize_ratio)),axis=1))
    
    #2 = log of nFr generalized
    matrix_generalized[:,2] = np.log(np.mean(matrix[0:int(length_generalized*generalize_ratio),10].reshape(-1,int(generalize_ratio)),axis=1))    

    return matrix_generalized

def plot_deterministic(matrix):
    """
    Function that plots the deterministic classification manually
    """
    fig, (ax1,ax2) = plt.subplots(1,2)
    fig.suptitle('normalized CPT data', fontsize=15)
    ax1.plot(np.log(matrix[:,9]),matrix[:,0], color='k')
    ax1.set_xlabel(r'ln(n$Q_t$) [-]', fontsize=13)
    ax1.set_ylabel('depth [m]', fontsize = 13)
    ax1.set_xlim(0,1.5*max(np.log(matrix[:,9])))
    ax1.invert_yaxis()
    ax1.set_xlim(0,max(np.log(matrix[:,9])+2))
    
    ax2.plot(np.log(matrix[:,10]),matrix[:,0],color='k')
    ax2.set_xlabel(r'ln(n$F_r$) [-]', fontsize=13)
    ax2.invert_yaxis()
    ax2.set_xlim(min(np.log(matrix[:,10])-2),max(np.log(matrix[:,10])+2))

    
    #add deterministic classification
    Tfont ={'fontname':'Times New Roman'}

    ax1.hlines(matrix[0,0],0,9, color='k', linewidth=0.5)
    ax1.hlines(3.5,0,7,color='k', linewidth=0.5)
    ax1.hlines(4,0,7,color='k', linewidth=0.5)
    ax1.hlines(4.9,0,7,color='k', linewidth=0.5)
    ax1.hlines(6.1,0,7,color='k', linewidth=0.5)
    ax1.hlines(6.9,0,9, color='k', linewidth=0.5)
    ax1.hlines(7.7,0,7,color='k', linewidth=0.5)
    ax1.hlines(8.05,0,7,color='k', linewidth=0.5)
    ax1.hlines(13.,0,7, color='k', linewidth=0.5)
    ax1.hlines(13.4,0,7, color='k', linewidth=0.5)
    ax1.hlines(13.9,0,9, color='k', linewidth=0.5)
    ax1.hlines(14.85,0,9, color='k', linewidth=0.5)
    ax1.hlines(15.55,0,7, color='k', linewidth=0.5)
    ax1.hlines(18.2,0,7,color='k',linewidth=0.5)
    ax1.hlines(18.6,0,7,color='k',linewidth=0.5)
    ax1.hlines(20,0,9, color='k', linewidth=0.5)
    ax1.hlines(20.6,0,7, color='k', linewidth=0.5)
    ax1.hlines(21.4,0,7,color='k', linewidth=0.5)
    ax1.hlines(24,0,7,color='k', linewidth=0.5)
    ax1.hlines(24.6,0,7,color='k', linewidth=0.5)

    ax1.hlines(matrix[-1,0],0,9, color='k', linewidth=0.5)
    ax1.vlines(7,matrix[0,0],matrix[-1,0], color='k', linewidth=1)
    
    ax1.fill_between(np.arange(0,7+1),matrix[0,0],3.5,facecolor=[0,0,0,0.7])
    ax1.fill_between(np.arange(0,7+1),3.5,4,facecolor=[0,0,0,0.9])
    ax1.fill_between(np.arange(0,7+1),4,4.9,facecolor=[0,0,0,0.7])
    ax1.fill_between(np.arange(0,7+1),4.9,6.1,facecolor=[0,0,0,0.9])
    ax1.fill_between(np.arange(0,7+1),6.1,6.9,facecolor=[0,0,0,0.7])
    ax1.fill_between(np.arange(0,7+1),6.9,7.7, facecolor=[0.7,0.4,0,1])
    ax1.fill_between(np.arange(0,7+1),7.7,8.05,facecolor=[0.6,0.6,0.2,1])
    ax1.fill_between(np.arange(0,7+1),8.05,13,facecolor=[0.7,0.4,0,1])
    ax1.fill_between(np.arange(0,7+1),13,13.4,facecolor=[0.6,0.6,0.2,1])
    ax1.fill_between(np.arange(0,7+1),13.4,13.9,facecolor=[0.7,0.4,0,1])
    ax1.fill_between(np.arange(0,7+1),13.9,14.85, facecolor=[0,0,0,0.9])
    ax1.fill_between(np.arange(0,7+1),14.85,15.55, facecolor=[0.6,0.6,0.2,1])
    ax1.fill_between(np.arange(0,7+1),15.55,18.2, facecolor=[0.7,0.4,0,1])
    ax1.fill_between(np.arange(0,7+1),18.2,18.6, facecolor=[0.6,0.6,0.2,1])
    ax1.fill_between(np.arange(0,8),18.6,20, facecolor=[0.7,0.4,0,1])
    ax1.fill_between(np.arange(0,7+1),20,20.6, facecolor=[0,0,0,0.7])
    ax1.fill_between(np.arange(0,7+1),20.6,21.4, facecolor=[0.6,0.6,0.2,1])
    ax1.fill_between(np.arange(0,7+1),21.4,24,facecolor=[0,0,0,0.7])
    ax1.fill_between(np.arange(0,7+1),24,24.6,facecolor=[0.6,0.6,0.2,1])
    ax1.fill_between(np.arange(0,7+1),24.6,matrix[-1,0],facecolor=[0,0,0,0.7])
    
    ax1.text(7.2,6,'Clay/',rotation=90, **Tfont, fontsize=12)
    ax1.text(7.7,4.5,'Peat',rotation=90, **Tfont, fontsize=12)

    ax1.text(7.3,11.5,'Sand',rotation=90, **Tfont, fontsize=14)
    ax1.text(7.3,14.8,'Peat',rotation=0, **Tfont, fontsize=10)
    ax1.text(7.2,19.2,'Sand-',rotation=90, **Tfont, fontsize=10)
    ax1.text(7.6,19.2,'mixtures',rotation=90, **Tfont, fontsize=10)
    ax1.text(7.3,23.7,'Clay',rotation=90, **Tfont, fontsize=14)
    plt.savefig('cpt_deterministic.pdf')

def plot_boundary_constraints(matrix,params):
    """
    Function that plots normalized data and boundary constraints. Function also checks if boundary conditions are valid
    """
    Tfont ={'fontname':'Times New Roman'}

    if params["implement constraints"] == "yes":
        
        boundary_constraint = params["boundary constraint"]
    if params ["implement constraints"] =="no":
        boundary_constraint=[]
    N_max = params["N max"]
    #check if boundary constraints are valid
    if len(boundary_constraint) >0:
        if max(boundary_constraint[:,-1]) >=N_max:
            raise ValueError("constraint given for boundary that exceeds N_max")
    
        for ii in range(1,N_max):
            if np.count_nonzero(boundary_constraint == ii) >ii:
                raise ValueError("too many boundary constraints given for N=",ii)
                
    fig, (ax1,ax2) = plt.subplots(1,2)
    fig.suptitle('normalized CPT data - constraints',**Tfont, fontsize=15)
    ax1.plot(np.log(matrix[:,9]),matrix[:,0], color='k')
    ax1.set_xlabel(r'ln(n$Q_t$) [-]', **Tfont, fontsize=13)
    ax1.set_ylabel('depth [m]', **Tfont, fontsize=13)
    ax1.set_xlim(0,1.5*max(np.log(matrix[:,9])))
    ax1.set_ylim(0,matrix[-1,0]+1)

    ax1.invert_yaxis()
    ax1.set_xlim(0,max(np.log(matrix[:,9])+2))
    ax2.plot(np.log(matrix[:,10]),matrix[:,0],color='k')
    ax2.set_ylim(0,matrix[-1,0]+1)
    ax2.set_xlabel(r'ln(n$F_r$) [-]', **Tfont, fontsize=13)
    ax2.invert_yaxis()
    ax2.set_xlim(min(np.log(matrix[:,10])-2),max(np.log(matrix[:,10])+2))
    for ii in range(len(boundary_constraint)):
        ax1.fill_between(np.arange(0,12),boundary_constraint[ii,1],boundary_constraint[ii,0], edgecolor='red', facecolor='red',alpha=1-boundary_constraint[ii,2]/N_max)
        ax2.fill_between(np.arange(-10,10),boundary_constraint[ii,1],boundary_constraint[ii,0], edgecolor='red', facecolor='red',alpha=1-boundary_constraint[ii,2]/N_max)
    plt.savefig('constraints.pdf')
    plt.show()
        
def collor_fill(layer_type):
    if layer_type == 1:
        collor_matrix = [0,0,0,0.2]
    if layer_type ==2:
        collor_matrix = [0,0,0,0.8]
    if layer_type ==3:
        collor_matrix = [0,0,0,0.7]
    if layer_type == 4:
        collor_matrix = [0.3,0.4,0.2,1]
    if layer_type == 5:
        collor_matrix = [0.6,0.6,0.2,1]
    if layer_type == 6:
        collor_matrix = [0.7,0.4,0,1]
    if layer_type == 7:
        collor_matrix = [0.7,0.2,0,1]
    if layer_type == 8:
        collor_matrix = [0.1,0.4,0.5,1]
    if layer_type == 9:
        collor_matrix = [0.1,0.4,0.5,0.8]   
    return collor_matrix
        

def plot_summary(matrix,summary,final_result,params):
    Tfont ={'fontname':'Times New Roman'}

    fig, (ax1,ax2) = plt.subplots(1,2)
    fig.suptitle('normalized CPT data', **Tfont, fontsize=20)
    ax2.plot(np.log(matrix[:,9]),matrix[:,0], color='k')
    ax2.set_xlabel(r'ln(n$Q_t$)', fontsize=15)
    ax1.set_ylabel('depth [m]')
    ax2.set_xlim(0,1.5*max(np.log(matrix[:,9])))
    ax2.set_ylim(0,matrix[-1,0]+1)
    ax2.invert_yaxis()
    ax2.set_xlim(0,max(np.log(matrix[:,9])+2))
    start = matrix[0,0]
    
    ax1.set_xlim(0,1)
    ax1.set_ylim(0,matrix[-1,0]+1)
    ax1.invert_yaxis()
    ax1.hlines(matrix[-1,0],0,1,color='k')
    ax1.hlines(start,0,1,color='k')
    ax1.set_xticks([])
    
    for i in range(len(summary)):
        bounds = summary[i,1:i+2]
        ax1.vlines((1/len(summary))*i,start,matrix[-1,0],color='k')
        ax1.text((1/len(summary))*i+(1/len(summary))/2 -0.02,1.5,f'$M_{i+1}$', fontsize=7.5, **Tfont)
        for j in range(len(bounds)):
            ax1.hlines(sum(bounds[:j])+matrix[0,0],(1/len(summary))*i, (1/len(summary))*i+(1/len(summary)),color='k',linewidth=0.5)
    
    for i in range(len(final_result)):
        collor_matrix = collor_fill(final_result[i,-1])
        if i == 0:
            ax2.fill_between(np.arange(0,10),start, start+final_result[0,0], facecolor=collor_matrix)
        else:
            ax2.fill_between(np.arange(0,10),start+sum(final_result[:i,0]),start+sum(final_result[:i+1,0]), facecolor=collor_matrix)
   # plt.savefig('summary_depth.pdf')
    plt.show()
    
    
    fig, ax = plt.subplots(figsize = (10,5))
    #domain
    ax.set_xlim(0,4+len(summary)*0.8+0.8)
    ax.set_ylim(-0.2-0.12*len(summary),1)
    #lines
    ax.hlines(0,0,1011.5, color='k',linewidth=1.5)
    ax.hlines(0.6,0,11.5, color='k',linewidth=1.5)
    ax.hlines(0.3,4,11.5, color='k', linewidth=1.5)
    
    #text
    ax.text(0.1,0.17,f'Model', **Tfont, fontsize=15)
    ax.text(0.1,0.07,f'Class $M_N$', **Tfont, fontsize=15)
    ax.text(1.7,0.07,r'ln[P(${\xi}$|$M_N$)]', **Tfont, fontsize=15)
    ax.text(4,0.5,'Most probable',**Tfont, fontsize=15)
    ax.text(4,0.37,r'thickness, $h^*_N$ (m)', **Tfont, fontsize=15)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    for i in range(len(summary)):
        ax.text(4+0.8*i,0.07,f"$h^*_{i+1}$", **Tfont, fontsize=15)
        ax.text(0.1,-0.1-i*0.12,f"$M_{i+1}$", **Tfont, fontsize=15)
        ax.text(1.7,-0.1-i*0.12,f"{round(summary[i,-1],1)}", **Tfont, fontsize=15)
        for j in range(len(summary[i])-2):
            value = round(summary[i,j+1],2)
            if value != 0:
                ax.text(4+j*0.8,-0.1-i*0.12,f'{value}', **Tfont, fontsize=15)
            else:
                ax.text(4+j*0.8,-0.1-i*0.12,f'  -', **Tfont, fontsize=15) 
  #  plt.savefig('summary_table.pdf')
    





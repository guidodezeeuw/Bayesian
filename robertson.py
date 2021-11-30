#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 10:58:23 2021

@author: guidodezeeuw
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 13:37:30 2021

@author: guidodezeeuw
"""

from shapely import geometry as geom
import numpy as np
import matplotlib.pyplot as plt
import descartes as des
import random


def plot_Robertson(matrix_generalized):
    """
    Get polygons
    """    
    largest_nFr = max((matrix_generalized[:,2]))
    #intersection points
    A = [-2.3026,0]
    B = [0.6569,0]
    C = [-2.3026,2.2268]       
    D = [2.3026,0]    
    E = [2.3026,2.0234]
    F = [0.5589,0.1776]
    G = [2.3026,3.9639]
    H = [1.8687, 4.0953]
    J = [1.4505,4.5104]
    K = [-1.3334,2.2126]
    L = [0.9622,5.3534]
    M = [-2.3026,3.4335]
    N = [0.3655,6.9078]
    O = [0.1658,6.9078]
    P = [-2.3026,5.0557]
    Q = [-2.3026,6.9078]
    R = [2.3026, 6.9078]
    S = [1.6334,6.9078]
    T = [-0.5773,1.7179]
    #if Fr values lie outside of given coordinates, extrapolate chart. 
    if largest_nFr>2.3026:
        D = [largest_nFr,0]
        E = [largest_nFr,0.5586*largest_nFr**2+-0.5399*largest_nFr+0.3049]
        G = [largest_nFr,0.8095*largest_nFr**2+-3.6795*largest_nFr+8.1444]
        R = [largest_nFr,6.9078]
    intersect = np.array([A,B,C,D,E,
                          F,G,H,J,K,
                          L,M,N,O,P,
                          Q,R,S,T])
    
    #x-range of functions
    I_range =np.arange(C[0],B[0],0.1)
    I_range1 = np.arange(C[0],K[0],0.1)
    I_range2 = np.arange(K[0],T[0],0.1)
    I_range3 = np.arange(T[0],F[0],0.1)
    I_range4 = np.arange(F[0],B[0],0.1)
    II_range =np.arange(F[0],E[0],0.1)
    III_range =np.arange(T[0],H[0],0.1)
    IV_range =np.arange(K[0],J[0],0.1)
    V_range =np.arange(M[0],L[0],0.1)
    VI_range =np.arange(P[0],O[0],0.1)
    VII_range =np.arange(N[0],G[0],0.1)
    VII_range1 = np.arange(N[0],L[0],0.1)
    VII_range2 = np.arange(L[0],J[0],0.1)
    VII_range3 = np.arange(J[0],H[0],0.1)
    VII_range4 = np.arange(H[0],G[0],0.1)
    VIII_range =np.arange(J[0],S[0],0.01)
    
    #y-values based on x-range
    I = -0.3707*I_range**2+-1.3625*I_range+1.0549
    I1 = -0.3707*I_range1**2+-1.3625*I_range1+1.0549
    I2 =-0.3707*I_range2**2+-1.3625*I_range2+1.0549
    I3 = -0.3707*I_range3**2+-1.3625*I_range3+1.0549
    I4 =-0.3707*I_range4**2+-1.3625*I_range4+1.0549
    II = 0.5586*II_range**2+-0.5399*II_range+0.3049
    III = 0.5405*III_range**2+0.2739*III_range+1.6959
    IV = 0.3833*IV_range**2+0.7805*IV_range+2.5718
    V = 0.2827*V_range**2+0.967*V_range+4.1612
    VI = 0.3477*VI_range**2+1.4933*VI_range+6.6507
    VII = 0.8095*VII_range**2+-3.6795*VII_range+8.1444
    VII1 = 0.8095*VII_range1**2+-3.6795*VII_range1+8.1444
    VII2 = 0.8095*VII_range2**2+-3.6795*VII_range2+8.1444
    VII3 = 0.8095*VII_range3**2+-3.6795*VII_range3+8.1444
    VII4 = 0.8095*VII_range4**2+-3.6795*VII_range4+8.1444
    VIII = 64.909*VIII_range**2+-187.07*VIII_range+139.2901
    

    #make lists for polygons
    #P1
    P1_X=np.concatenate((I_range, [B[0],A[0]]))
    P1_Y=np.concatenate((I,[B[1],A[1]]))
    P1_mat = np.column_stack((P1_X,P1_Y))
    #P2
    P2_X=np.concatenate((II_range,[E[0],D[0],B[0]],np.flip(I_range4)))
    P2_Y=np.concatenate((II,[E[1],D[1],B[1]],np.flip(I4)))
    P2_mat = np.column_stack((P2_X,P2_Y))
    
    #P3
    P3_X = np.concatenate((III_range,VII_range4,[G[0],E[0]],np.flip(II_range),np.flip(I_range3[1:])))
    P3_Y = np.concatenate((III,VII4,[G[1],E[1]],np.flip(II),np.flip(I3[1:])))
    P3_mat = np.column_stack((P3_X,P3_Y))
    
    #P4
    P4_X = np.concatenate((IV_range,VII_range3,[H[0]],np.flip(III_range),[T[0]],np.flip(I_range2)))
    P4_Y = np.concatenate((IV,VII3,[H[1]],np.flip(III),[T[1]],np.flip(I2)))
    P4_mat = np.column_stack((P4_X,P4_Y))
    
    #P5
    P5_X = np.concatenate((V_range,VII_range2,[J[0]],np.flip(IV_range),[K[0]],np.flip(I_range1)))
    P5_Y = np.concatenate((V,VII2,[J[1]],np.flip(IV),[K[1]],np.flip(I1)))
    P5_mat = np.column_stack((P5_X,P5_Y))
    
    #P6
    P6_X = np.concatenate((VI_range,[O[0]],VII_range1,[L[0]],np.flip(V_range)))
    P6_Y = np.concatenate((VI,[O[1]],VII1,[L[1]],np.flip(V)))
    P6_mat = np.column_stack((P6_X,P6_Y))
    
    #P7
    P7_X = np.concatenate(([Q[0],O[0]],np.flip(VI_range)))
    P7_Y = np.concatenate(([Q[1],O[1]],np.flip(VI)))
    P7_mat = np.column_stack((P7_X,P7_Y))
    
    #P8
    P8_X = np.concatenate(([S[0]],np.flip(VIII_range),np.flip(VII_range2),np.flip(VII_range1)))
    P8_Y = np.concatenate(([S[1]],np.flip(VIII),np.flip(VII2),np.flip(VII1)))
    P8_mat = np.column_stack((P8_X,P8_Y))
    
    #P9
    P9_X = np.concatenate(([S[0],R[0],G[0]],np.flip(VII_range4), np.flip(VII_range3),VIII_range[1:]))
    P9_Y = np.concatenate(([S[1],R[1],G[1]],np.flip(VII4),np.flip(VII3),VIII[1:]))
    P9_mat = np.column_stack((P9_X,P9_Y))
    
    """
    Plot Polygons
    """
    fig = plt.figure(figsize=(5,8))
    ax = fig.add_subplot(111)
    
    #add polygons to plot
    ax.add_patch(des.PolygonPatch(geom.Polygon(P1_mat), facecolor=[0,0,0,0.2], label='1. Sensitive fine-grained'))
    ax.add_patch(des.PolygonPatch(geom.Polygon(P2_mat), facecolor=[0,0,0,0.8], label = '2. Organic'))
    ax.add_patch(des.PolygonPatch(geom.Polygon(P3_mat), facecolor=[0,0,0,0.7], label= '3. Clay'))
    ax.add_patch(des.PolygonPatch(geom.Polygon(P4_mat), facecolor=[0.3,0.4,0.2,1], label='4. Silt-mixtures'))
    ax.add_patch(des.PolygonPatch(geom.Polygon(P5_mat),facecolor=[0.6,0.6,0.2,1], label='5. Sand-mixtures'))
    ax.add_patch(des.PolygonPatch(geom.Polygon(P6_mat), facecolor=[0.7,0.4,0,1], label='6. Sand'))
    ax.add_patch(des.PolygonPatch(geom.Polygon(P7_mat), facecolor=[0.7,0.2,0,1], label='7. Gravelly sand to sand'))
    ax.add_patch(des.PolygonPatch(geom.Polygon(P8_mat), facecolor=[0.1,0.4,0.5,1], label='8. Very stiff sand to clayey sand'))
    ax.add_patch(des.PolygonPatch(geom.Polygon(P9_mat), facecolor=[0.1,0.4,0.5,0.8], label='9. Very stiff fine-grained'))
    
    #plot specifics
    ax.set_xlim(A[0],D[0])
    ax.set_ylim(A[1],Q[1])
    #ax.set_xlim(-2.3026,2.3026)
    leg = ax.legend(bbox_to_anchor=(1.01,1),title='Legend', title_fontsize=15)
    leg._legend_box.align = "left"
    ax.text(-1.5,1,'1',fontsize=15)
    ax.text(1.8,0.5,'2',fontsize=15)
    ax.text(1.2,1.7,'3',fontsize=15)
    ax.text(0.6,2.6,'4',fontsize=15)
    ax.text(0,3.3,'5',fontsize=15)
    ax.text(-0.8,4.5,'6',fontsize=15)
    ax.text(-1.5,6,'7',fontsize=15)
    ax.text(1.05,6,'8',fontsize=15)
    ax.text(1.9,5,'9',fontsize=15)
    ax.set_title('normalized Robertson chart', fontsize=20)
    ax.set_xlabel(r'ln$(nF_r)$', fontsize=15)
    ax.set_ylabel(r'ln$(nQ_t)$', fontsize=15)    

    Polygons = np.array([geom.Polygon(P1_mat),geom.Polygon(P2_mat),geom.Polygon(P3_mat),
                        geom.Polygon(P4_mat),geom.Polygon(P5_mat),geom.Polygon(P6_mat),
                        geom.Polygon(P7_mat),geom.Polygon(P8_mat),geom.Polygon(P9_mat)])
    
    ax.plot(matrix_generalized[:,2], matrix_generalized[:,1], marker='o',ls='',color='k',markersize=2)
    return Polygons

def MC_probability(matrix_generalized, Polygons, params):
    std_Fr = params["std Fr"]
    std_Qt = params["std Qt"]
    iterations = params["probability iterations"]
    for jj in range(len(matrix_generalized)):
        data = (matrix_generalized[jj,2],matrix_generalized[jj,1])
        #ax.plot(data[0],data[1],marker='o',ls='')
    
        std_Fr = 1
        std_Qt = 1.2
        print(jj)
        count = np.array([0,0,0,0,0,0,0,0,0])
        for ii in range(0,iterations):
            
            Fr =random.gauss(data[0],std_Fr)
            Qt =random.gauss(data[1],std_Qt)
            point = geom.Point(Fr,Qt)
            if point.within(Polygons[0]) == True:
                count[0] += 1
            if point.within(Polygons[1]) == True:
                count[1] += 1
            if point.within(Polygons[2]) == True:
                count[2] += 1
            if point.within(Polygons[3]) == True:
                count[3] += 1
            if point.within(Polygons[4]) == True:
                count[4] += 1
            if point.within(Polygons[5]) == True:
                count[5] += 1
            if point.within(Polygons[6]) == True:
                count[6] += 1
            if point.within(Polygons[7]) == True:
                count[7] += 1
            if point.within(Polygons[8]) == True:
                count[8] += 1
        matrix_generalized[jj,3:] = count/np.sum(count)

    return matrix_generalized







#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 12:55:13 2021

@author: guidodezeeuw
"""
import pandas as pd
import numpy as np
import robertson as Frob
import read_cpt as Fread
import posterior as Fpost
import os 
import matplotlib.pyplot as plt

"""
####################################
#-------------1. Input-------------#
####################################
"""

params = {
    "cpt":              "CPT000000097976_IMBRO_A.gef",          #cpt gef file
    "N max":            6,                                      #maximum soil layers to consider
    "min thickness":    0.1,                                    #minim thickness of a layer to be considered a separate layer
    "std Fr":           1,                                      #standard deviation of friction ratio
    "std Qt":           1.2,                                    #standard deviation of cone resistance
    "implement constraints": "no",                             #if desired to use constraints: "yes". if no constraints are considered "no"                       
    "boundary constraint": np.array([[6,9,2],                   #specify constraints
                                     [19.5,20.2,3],
                                     [15,16,4]]),
    "probability iterations": 200000,                          #number of iterations to get probability per measurement
    "treshold model class": 4,                                  #largest model class that does not need an optimization
    "no. guesses": 20,                                           #number of initial guesses used for optimization per model class
    "guesses iterations": 400000                                #number of iterations of which no. guesses are stored
    }

"""
####################################
#------------2a. read cpt----------#
#--------2b. generalize cpt--------#
####################################
"""
#2a.
matrix = Fread.read_cpt(params)                                            #read cpt file and store in matrix
Fread.plot_boundary_constraints(matrix,params)       #plot figure with boundary constraints to check

#2b.
matrix_generalized = Fread.generalize(matrix,params)             #generalize matrix based on min_thickness

Frob.plot_Robertson(matrix_generalized)
print("Obtain matrix: SUCCES")

"""
####################################
#--------3 get probability per data point---------#
####################################
"""

Polygons = Frob.plot_Robertson(matrix_generalized)                      #get polygons from robertson chart
if os.path.isfile('prob_generalized2.csv') == False:                     #check if file of probability already exists. if not: make one
    matrix_generalized = Frob.MC_probability(matrix_generalized,Polygons,params)   #get probability per measurement
    np.savetxt('prob_generalized2.csv', matrix_generalized,delimiter=";",header='depth;nQt;nFr;P_1;P_2;P_3;P_4;P_5;P_6;P_7;P_8;P_9')    

if os.path.isfile('prob_generalized2.csv') == True:
    matrix_generalized = np.array(pd.read_csv('prob_generalized2.csv', sep=';')) 

print("get probability per datapoint: SUCCES")

"""
####################################
#---------4. calculate most probable N and layer thicknesses-----------#
####################################
"""
final_result,summary= Fpost.calculator(matrix_generalized,params)

"""
####################################
#---------5. plot results-----------#
####################################
"""
Fread.plot_summary(matrix,summary,final_result,params)


 
    


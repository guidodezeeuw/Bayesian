#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 15:04:04 2021

@author: guidodezeeuw
"""

import numpy as np
import random
import scipy.optimize as optimize
import math

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

"""
#############################
------HELPFUL FUNCTION-------
#############################
"""

def find_nearest(array,value):
    """
    function to calculate index in array closest to the value given
    """
    idx=[]
    for ii in value:
        idx.append(np.abs(array-ii).argmin())
    return idx


def thickness_to_boundaries(thickness,matrix_generalized):
    """
    Function to get the indices of the boundaries (convert array with thicknesses information to array with depth (boundary) information)
    """
    boundaries = np.zeros(len(thickness)+1)
    
    boundaries[0] = matrix_generalized[0,0]     #first boundary is equal to the top of soil profile
    for ii in range(len(boundaries)-1):
        boundaries[ii+1] = matrix_generalized[0,0] + sum(thickness[:ii+1])
    
    """ 2.1 get indices in data matrix"""
    soil_boundaries = np.arange(0,len(thickness)+1)   
    indices = find_nearest(matrix_generalized[:,0],boundaries[soil_boundaries])
    indices=np.array(indices) #index positions of boundaries in data matrix.  
    
    return indices

def Eqs_Wang(matrix_generalized,indices):
    """
    function that follows Eq. 2 and 1 given by Wang et al. (2013). calculates, for each layer, the chance that that layer belongs fully to 1 single soil type
    """
    P_ST_J=np.zeros((len(indices)-1,9))                            #matrix in which results following from eq. 2 (Wang et al., 2013) are stored
    P_eps_n=np.zeros(len(indices)-1)
                               #matrix in which results following from eq. 1 (Wang et al., 2013) are stored
    """ 2.2 eq. 2 & 1 Wang et al. (2013) """
    #the subdivision between upper and lower bound has no coding benefit other than being more clear to the reader.
    upper_bound = indices[0:-1] #upper bounds of soil layers
    lower_bound = indices[1:] #lower bounds of soil layers
    
    for nn in range(0,len(indices)-1): #iterates for each soil layer in the model state (with different N)
        P_ST_J[nn,0]= np.sum(np.log(matrix_generalized[upper_bound[nn]:lower_bound[nn],3])) #for soil type 1, Eq. 2
        P_ST_J[nn,1]= np.sum(np.log(matrix_generalized[upper_bound[nn]:lower_bound[nn],4])) #for soil type 2, Eq. 2
        P_ST_J[nn,2]= np.sum(np.log(matrix_generalized[upper_bound[nn]:lower_bound[nn],5])) #for soil type 3, Eq. 2
        P_ST_J[nn,3]= np.sum(np.log(matrix_generalized[upper_bound[nn]:lower_bound[nn],6])) #for soil type 4, Eq. 2
        P_ST_J[nn,4]= np.sum(np.log(matrix_generalized[upper_bound[nn]:lower_bound[nn],7])) #for soil type 5, Eq. 2
        P_ST_J[nn,5]= np.sum(np.log(matrix_generalized[upper_bound[nn]:lower_bound[nn],8])) #for soil type 6, Eq. 2
        P_ST_J[nn,6]= np.sum(np.log(matrix_generalized[upper_bound[nn]:lower_bound[nn],9])) #for soil type 7, Eq. 2
        P_ST_J[nn,7]= np.sum(np.log(matrix_generalized[upper_bound[nn]:lower_bound[nn],10])) #for soil type 8, Eq. 2
        P_ST_J[nn,8]= np.sum(np.log(matrix_generalized[upper_bound[nn]:lower_bound[nn],11])) #for soil type 9, Eq. 2
        P_eps_n[nn] = np.sum(np.exp(P_ST_J[nn]))        #Eq. 1
        
    return P_eps_n

def soil_type(matrix_generalized, indices):
    """
    calculates most probable soiltype (1-9) per layer based on the given boundaries
    """
    P_ST_J=np.zeros((len(indices)-1,9))                            #matrix in which results following from eq. 2 (Wang et al., 2013) are stored
    soiltype_final = np.zeros((len(indices)-1))
    
    """ 2.2 eq. 2 & 1 Wang et al. (2013) """
    #the subdivision between upper and lower bound has no coding benefit other than being more clear to the reader.
    upper_bound = indices[0:-1] #upper bounds of soil layers
    lower_bound = indices[1:] #lower bounds of soil layers
    
    for nn in range(0,len(indices)-1): #iterates for each soil layer in the model state (with different N)
        P_ST_J[nn,0]= np.sum(np.log(matrix_generalized[upper_bound[nn]:lower_bound[nn],3])) #for soil type 1
        P_ST_J[nn,1]= np.sum(np.log(matrix_generalized[upper_bound[nn]:lower_bound[nn],4])) #for soil type 2
        P_ST_J[nn,2]= np.sum(np.log(matrix_generalized[upper_bound[nn]:lower_bound[nn],5])) #for soil type 3
        P_ST_J[nn,3]= np.sum(np.log(matrix_generalized[upper_bound[nn]:lower_bound[nn],6])) #for soil type 4
        P_ST_J[nn,4]= np.sum(np.log(matrix_generalized[upper_bound[nn]:lower_bound[nn],7])) #for soil type 5
        P_ST_J[nn,5]= np.sum(np.log(matrix_generalized[upper_bound[nn]:lower_bound[nn],8])) #for soil type 6
        P_ST_J[nn,6]= np.sum(np.log(matrix_generalized[upper_bound[nn]:lower_bound[nn],9])) #for soil type 7
        P_ST_J[nn,7]= np.sum(np.log(matrix_generalized[upper_bound[nn]:lower_bound[nn],10])) #for soil type 8
        P_ST_J[nn,8]= np.sum(np.log(matrix_generalized[upper_bound[nn]:lower_bound[nn],11])) #for soil type 9
        soiltype_final[nn] = np.argmax(P_ST_J[nn,:]) +1
    return soiltype_final    


"""
#############################
-------MAIN FUNCTIONS--------
#############################
"""

def objective_function(thickness,matrix_generalized):
    """
    calculates objective function = -ln(likelihood function)
    """
    
    indices = thickness_to_boundaries(thickness,matrix_generalized) #get indices of the layer boundaries in the matrix
    """ 2.2 eq. 2 & 1 Wang et al. (2013) """
    P_eps_n = Eqs_Wang(matrix_generalized,indices)
    """ 2.3 eq. 6 Wang et al. (2013) """
    P_eps = np.sum(np.log(P_eps_n))
    return -P_eps


def thickness_generator(matrix_generalized,model_class,min_thickness,boundary_constraint):
    
    boundaries = np.zeros(model_class+1)
    """ 0. make array of possible boundaries, spacing of 0.1 """
    boundary_possibilities = np.arange(round(matrix_generalized[0,0],1)+0.1*(round(matrix_generalized[0,0],1)<matrix_generalized[0,0])+min_thickness,
                                            round(matrix_generalized[-1,0],1)-0.1*(round(matrix_generalized[-1,0],1)>matrix_generalized[-1,0])+0.1,0.1)
    
    """ 1. get thickness distribution """
    #1.1 first boundary is start of profile (depth at which cpt starts)
    boundaries[0] = matrix_generalized[0,0]   

    #1.2 generate random values within constraint
    L_constraints=np.array([])
    for constraint in boundary_constraint:
        if model_class >= constraint[-1]:
            random_constraint = round(np.random.uniform(constraint[0],constraint[1]),1)
            L_constraints=np.append(L_constraints,random_constraint)
            idx = np.array(np.where((boundary_possibilities>random_constraint-min_thickness)           #indices of value +- min_thickness in boundary_possibilities
                                    *(boundary_possibilities<random_constraint+min_thickness))[0])
            boundary_possibilities=np.delete(boundary_possibilities,idx)                            #delete indices so that next iteration only a correct boundary

    #1.3 add constraints to boundaries
    boundaries[1:len(L_constraints)+1] = L_constraints
    #1.4 give random values to other boundaries (but not for first modelclass, as then there is only 1 layer)
    if model_class>1:
        for nn in range(1+len(L_constraints),model_class+1):
            boundaries[nn] = random.choice(boundary_possibilities)
            idx = np.array(np.where((boundary_possibilities>boundaries[nn]-min_thickness)           #indices of value +- min_thickness in boundary_possibilities
                                    *(boundary_possibilities<boundaries[nn]+min_thickness))[0])
            boundary_possibilities=np.delete(boundary_possibilities,idx)                            #delete indices so that next iteration only a correct boundary
            
    #1.3 last boundary will be at final depth of CPT
    boundaries[model_class] = matrix_generalized[-1,0] 
    
    boundaries=np.sort(boundaries)                      #sort boundary list.
    thickness =np.diff(boundaries)        
    
    return thickness


def best_guesses(matrix_generalized,model_class,params):

    """
    Function that calculates a number (=no_guesses) of best guesses. that is, for layer thickness configurations that give the lowest objective functions
    """
    min_thickness= params["min thickness"]
    boundary_constraint = params["boundary constraint"]
    no_guesses = params["no. guesses"]
    iterations = params["guesses iterations"]
    best_conf = np.zeros((no_guesses,model_class+1)) #1st column = posterior value, others = thicknesses
    
    for i in range(iterations):
        """ 1. obtain random thickness distribution"""
        thickness = thickness_generator(matrix_generalized,model_class,min_thickness,boundary_constraint)

        """ 2. get likelyhood function """
        P_eps = objective_function(thickness,matrix_generalized)
                
        """ 3. store best ones in matrix """
        if i < no_guesses:                      #first fill guess matrix (which is no all 0's))
            best_conf[i,0] = P_eps
            best_conf[i,1:]=thickness

        if i >= no_guesses:                     #then update guess matrix if a more preferred value is found
            pos = np.where(best_conf[:,0] == max(best_conf[:,0]))[0][0]
            if best_conf[pos,0] > P_eps:
                best_conf[pos,0] = P_eps
                best_conf[pos,1:] = thickness

    return best_conf


def optimizer(matrix_generalized,model_class, params,best_conf):
    """ 
    Function that further optimizes a layer thickness configuration based on the basin-hopping approach
    """
    min_thickness = params["min thickness"]     #minimal thickness to consider
    thickness_tot = round(matrix_generalized[-1,0]-matrix_generalized[0,0],2)  
    cons = [] #list summarizing the constraints that will be used by the optimizer
    
    cons.append({'type':'eq','fun': lambda thickness: np.sum(thickness)-thickness_tot}) #constraint 1 = sum of thicknesses must be equal to total thickness
    
    for con in range(model_class):
        cons.append({'type':'ineq','fun': lambda thickness: thickness[con]-min_thickness}) #constraint 2 = thickness must be larger or equal to the minimal thickness
    
    bounds = [] #list summarizing the bounds of the thickness
    for bound in range(model_class):
        bounds.append([0.2,thickness_tot]) #bound 1= thickness per layer must be minimal of 0.2 and maximum of total thickness
        
    minimizer_kwargs = dict(method='SLSQP',args=(matrix_generalized), constraints=cons, bounds=bounds, tol=0.01, options=dict(maxiter=200))
    
    min_objective = np.array([np.inf])    #an array where the minimal value of the objective function will be stored. start with infinity
    thicknessL = np.zeros((1,model_class))       

    for opt in range(len(best_conf)): #optimize for all best configurations
        initial_guess = best_conf[opt,1:] #specify initial guess for optimizer
        counter = 0 #start a counter
        while counter == 0: #as long as counter = 0, keep trying to optimize that configuration
            res_loop = optimize.basinhopping(objective_function,initial_guess,minimizer_kwargs=minimizer_kwargs,niter=500, niter_success=100)
            if res_loop.lowest_optimization_result.success == True: #if optimization is found, add counter so optimizing for that configuration will stop
                counter += 1
                if res_loop.fun < min_objective[0]: #if the new optimized value is lower than the one stored in "objective", change old value to this value
                    min_objective[0] = res_loop.fun #store current optimal objective function
                    thicknessL[:] = np.array(res_loop.x)[:] #store thicknesses for that optimal value
                    
            if res_loop.lowest_optimization_result.success == False: #if optimization failed, try again
                print('False optimization found... try again...')
    return min_objective, thicknessL #return minimized objective function and corresponding thicknesses
    

def calculator(matrix_generalized,params):
    """
    main function solving for the different sections
    """
    
    N_max = params["N max"]
    treshold = params["treshold model class"]   
    std_Fr = params["std Fr"]                                      #standard deviation of friction ratio
    std_Qt = params["std Qt"]                                      #standard deviation of cone resistance
    k = np.arange(1,N_max+1)

    """
    PART 1
    """
    
    """1. if a probability = 0, this will cause problems --> change all 0's to a small number"""
    
    thickness_tot = matrix_generalized[-1,0]-matrix_generalized[0,0]
    zero_position = np.argwhere(matrix_generalized[:,3:]==0)
    for index in zero_position:
        matrix_generalized[index[0],index[1]+3]=0.00001
    
    all_optimized = np.zeros((int(N_max),int(N_max)+1))     #list withb best layer thickness configuration per model class

    """
    PART 2: minimize objective function (=-ln(likelihood function)) per model class
    """
       
    """2a. Model class 1: having 1 layer, so only 1 solution"""
    
    N1 = objective_function([thickness_tot],matrix_generalized) #calculate objective function for model class 1--> 0 boundaries
    all_optimized[0,0:2]=[N1,thickness_tot]
    print("Caclulation for Model Class 1: SUCCES")


    """2b. calculate for other model classes"""
    
    for N in range(1+1,N_max+1): #for every model class
        
        """first, get best number (=no_guesses) of best guesses"""     
        best_conf= best_guesses(matrix_generalized,N,params)

        """if N is lower than specified number of layers (=threshold), no optimization is needed--> take lowest of best guesses"""
        if N<=treshold:
            min_index = np.argmin(best_conf[:,0])
            all_optimized[N-1,:N+1]=best_conf[min_index]
            
        """if N is larger than specified number of layers (=threshold), optimization is needed."""
        if N>treshold:
            objective, thicknessL = optimizer(matrix_generalized,N,params,best_conf) #optimizes for each guess and returns most optimal optimized value
            all_optimized[N-1,0]=objective
            all_optimized[N-1,1:N+1]=thicknessL
        
        """check if two subsequent layers have the same soil type"""
        thickness_check = all_optimized[N-1,1:N+1]
        indices_check = thickness_to_boundaries(thickness_check,matrix_generalized)
        soiltype_check = soil_type(matrix_generalized,indices_check) #list of soil types for best configuration
        for ii in range(len(soiltype_check)-1):
            if soiltype_check[ii+1]==soiltype_check[ii]:
                bottom_mask = np.sum(thickness_check[:ii])+matrix_generalized[0,0]
                top_mask = np.sum(thickness_check[:ii+2])+matrix_generalized[0,0]
                mask=[bottom_mask,top_mask]
                
                """further add this part"""
                #1. get new best_conf. using best_guesses
                #but now, add the parameter mask. then under "1. obtain random thickness distribution"
                #use this mask to exclude values between these two (bottom_mask and top_mask) boundaries
                #this results in a list of best_conf with layer boundaries NOT between the masked boundaries
                #this is done simply by removing the values from the boundary_possibilities list. for example:
                #for ii in range(len(boundary_possibilities)):
                    #if mask[0] >= boundary_possibilities[ii] <= mask[1]:
                        #np.delete(boundary_possibilities,boundary_possibilities[ii])
                
                #2. then, follow every step without any more changes
                #however, and this is why i did not implement this myself, this adjustment has some errors regarding consistency
                #error 1: the specified constraints at the start of the simulation might interfere with the applied mask. 
                #for example, a mask might exclude depth 5 m- 10 m from the new analysis, but a constraint is given
                #between 7-8 m. this is not possible, so maybe, constraints that fall within that mask should be neglected
                
                #error 2: what if there are three subsequent layers with similar soil types? one should account for this!
                #for example, layer 2,3 and 4 have soil type 5
                
                #error 3: what if there are multiple subsequent layers with similar soil types?
                #for example, layer 2 and 3 have similar soil types, but also layer 5 and 6
                #this should be implemented as well
                
                #error 4: what if the newly generated soil types have the same problem? maybe make a function
                #that loops and further simplifies the soil profile (by adding more masks) as long as similar soil types are found
                #you could add a condition.
                #for example, a pseudocode:
                    
                #condition = 0
                #if condition <1:
                                      
                    #1. calculate thicknesses and soil types
                    
                    #2. reflect on condition
                    
                    #if condition is met (=no subsequent soil layers with similar soil type):
                        #condition += 1
                    
                    #else (=subsequent soil layers with similar soil type is found):
                        #condition += 0
                        #add mask
                        #do loop again
                        
        print(f"Calculation for Model Class: {N}: SUCCES")


    """PART 3: calculate most probable model class"""   
    
    likelihood = -all_optimized[:,0]
    prior_distribution = np.log((1/std_Fr * 1/std_Qt)**k)
    
    
    #3a. calculate conditional probability (Eq. 12, Wang et al. (2013))
    cond_prob = likelihood+prior_distribution
    cond_prob = np.exp(cond_prob)
    cond_prob = np.log(cond_prob/(thickness_tot**(k-1)))
    
    #3b. maximize posterior function and select layer and thicknesses
    max_index = np.argmax(cond_prob)
    final_class = all_optimized[max_index,:max_index+2]
    
    """PART 4: get soil types for that model class"""
    
    #get soil types for final class
    thickness_final = final_class[1:]
    indices = thickness_to_boundaries(thickness_final,matrix_generalized)
    soiltype_final = soil_type(matrix_generalized,indices) 

    """PART 5: save results"""
    #get summary
    cond_prob = np.reshape(cond_prob,(len(cond_prob),1))
    summary = np.hstack((all_optimized,cond_prob))
    
    #get final configuration
    final_result = np.column_stack([thickness_final, soiltype_final])

    return final_result, summary
        
        
    
    
    
    
    
    
    
    
    
    

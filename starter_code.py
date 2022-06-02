# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 10:48:30 2022

@author: fhu14
"""
'''
Starter code for generating/reproducing the heatmap by comparing different methods.
For more complete documentation, refer to the functions defined in the module
ani1_interface
'''

#%% Imports, definitions
import pickle
from ani1_interface import get_ani1data
from typing import List, Dict
import numpy as np
from collections import Counter
Array = np.ndarray


#%% Code behind

def fit_linear_ref_ener(dataset: List[Dict], target1: str,
                        target2: str, allowed_Zs: List[int]) -> Array:
    r"""Fits a linear reference energy model between the DFTB+ method and some
        energy target
        
    Arguments:
        dataset (List[Dict]): The list of molecule dictionaries that have had the
            DFTB+ results added to them.
        target1 (str): The starting point energy target
        target2 (str): The second energy target that you need to correct for
        allowed_Zs (List[int]): The allowed atoms in the molecules
    
    Returns:
        coefs (Array): The coefficients of the reference energy
        XX (Array): 2D matrix in the number of atoms
        
    Notes: The reference energy corrects the magnitude between two methods
        in the following way: 
            
        E_2 = E_1 + sum_z N_z * C_z + C_0
        
        where N_z is the number of times atom z occurs within the molecule and 
        C_z is the coefficient for the given molecule. This is accomplished by solving
        a least squares problem.
        
        The reference energy vector is generated through a matrix multiply. 
        Suppose that E_1 is the vector of energies for the molecules in the 
        given dataset. The corrected energies, E_corrected, is generated as follows:
        
        E_corrected = E_1 + (XX @ coefs)
        
        where XX and coefs are the output of this function.
    """
    nmol = len(dataset)
    XX = np.zeros([nmol, len(allowed_Zs) + 1])
    method1_mat = np.zeros([nmol])
    method2_mat = np.zeros([nmol])
    iZ = {x : i for i, x in enumerate(allowed_Zs)}
    
    for imol, molecule in enumerate(dataset):
        Zc = Counter(molecule['atomic_numbers'])
        for Z, count in Zc.items():
            XX[imol, iZ[Z]] = count
            XX[imol, len(allowed_Zs)] = 1.0
        method1_mat[imol] = molecule['targets'][target1] 
        method2_mat[imol] = molecule['targets'][target2]
    
    yy = method2_mat - method1_mat
    lsq_res = np.linalg.lstsq(XX, yy, rcond = None)
    coefs = lsq_res[0]
    return coefs, XX

'''
Consider the following workflow
    1) Use get_ani1data and the ani1 h5 data file (found on Canvas) to generate a series of 
        molecules from the overall ANI-1ccx dataset
    2) Save the molecules using pickle
    3) Calculate differences and save those too (This will involve using the above function for 
        generating linear reference energy parameters)
    4) Generate the heatmap using seaborn
    5) Think about outlier rejection

I recommend that you guys work collaboratively on this, whether that is dividing
the work into different functions or programming together. Again, the method of
working is up to your group.
'''






#%% Main block
'''
For organization, I would keep functions separate from top-level code to be executed.
For that reason, define functions in the 'Code behind' block of the file and 
write executable code in the 'Main block' portion.
'''
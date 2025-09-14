# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:10:11 2024

@author: shreyas
"""

import h5py
import numpy as np
import pickle
from numpy.random import uniform
from scipy.linalg import expm
from scipy.optimize import minimize
from sys import exit, argv
import time
       
path = ['einsum_path', (1, 3), (1, 2), (0, 1), (0, 1)]

def num_params(Nl,L):
    c = Nl**2 * L * (L-1) // 2
    u = L * Nl * (Nl - 1) // 2

    return c, u, c + u

def construct_cartan_tensor(lam, Nl, L):

    tbt = np.zeros([L,L,Nl,Nl,Nl,Nl])

    tally = 0
    for i in range(L):
        for j in range(i):
            for p in range(Nl):
                for q in range(Nl):
                    # j >= i  terms will be zero.
                    # switch i,j & switch p,q to get same tensor (not here)
                    tbt[i,j,p,q,p,q] += lam[tally]
                    #tbt[j,i,q,p,q,p] += lam[tally]
                    tally            += 1

    return tbt

def construct_orthogonal(theta, Nl, L):

    X = np.zeros([L, Nl, Nl])

    tally = 0
    for i in range(L):
        for p in range(Nl):
            for q in range(p+1, Nl):
                X[i,p,q] += theta[tally]
                X[i,q,p] -= theta[tally]
                tally    += 1

    return np.array([expm(X[i,:,:]) for i in range(L)])

def get_fragment(x, Nl, L):
    c, u, p = num_params(Nl, L)

    
    lam    = x[ : c]
    theta  = x[c : ]
    #print('# of parameters in lambda', lam.size, '# of parameters in theta',theta.size)

    tbt = construct_cartan_tensor(lam, Nl, L)
    O   = construct_orthogonal(theta, Nl, L)

    return np.einsum('lmpqpq,lpa,mqb,lpc,mqd->lmabcd', tbt, O, O, O, O,optimize=path)
  

def evaluate_cost_function(x, target_tbt, Nl, L):
    fragment_tbt = get_fragment(x, Nl, L)
    diff         = (fragment_tbt - target_tbt)

    return np.sum(diff * diff)

def obtain_gfro_fragment(target_tbt):
    # Number  of modes and modals and parameter counts
    L       = target_tbt.shape[0]
    Nl      = target_tbt.shape[2]
    c, u, p = num_params(Nl, L)

    # cost function
    def cost(x):
        return evaluate_cost_function(x, target_tbt, Nl, L)

    # initial guess
    lam      = np.zeros(c)
    theta    = np.array([uniform(-np.pi/2, np.pi/2, Nl*(Nl-1)//2) for i in range(L)]).reshape(u)
    x0       = np.concatenate((lam,theta))
    
    # options
    options = {
        'maxiter' : 1000,
        'disp'    : False,
    }

    #tolerance
    #tol     = 1e-3
    #enum    = L * (L-1) * Nl ** 4
    #fun_tol = (tol / enum) ** 2
    
        
    def printx(xn):
	    with open('Fragments.out', 'a') as f:
    		print(f'cost : {cost(xn)}\n', file=f)

    # optimize
    return minimize(cost, x0, method='BFGS', options=options,tol=5e-1,callback=printx)

#save fragments
def save_params(mol, nm, params, norms):
    filename = mol+ '_VSCFfragments_' + str(nm) + 'modals.out'
    with open(filename, 'wb') as f:
        pickle.dump([params, norms], f)
    return None

#Load in fragments that were found in earlier calculation
def load_params(mol, nm):
    filename = mol+ '_VSCFfragments_' + str(nm) + 'modals.out'
    with open(filename, 'rb') as f:
        params, norms = pickle.load(f)
    return params, norms



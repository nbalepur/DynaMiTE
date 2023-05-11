#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#utility functions for running the CD method
#loss: min 1/2 \sum_t | Yt - UtVt' |^2 + lam/2 \sum_t(|Ut|^2 + |Vt|^2) + 
#                                        tau/2 \sum_t>1(|Vt - Vt-1|^2 + |Ut - Ut-1|^2)
#                                        gam/2 \sum_t (|Ut - Vt|^2)

import numpy as np
import scipy.io as sio
import copy

def update_fast(U,Y,Vm1,Vp1,lam,tau,gam,ind,iflag):
    
    UtU = np.dot(U.T,U) # rxr
    r = UtU.shape[0]    
    if iflag:   M   = UtU + (lam + 2*tau + gam)*np.eye(r)
    else:       M   = UtU + (lam + tau + gam)*np.eye(r)
       
    Uty = np.dot(U.T,Y) # rxb
    Ub  = U[ind,:].T   # rxb
    A   = Uty + gam*Ub + tau*(Vm1.T+Vp1.T)  # rxb
    Vhat = np.linalg.lstsq(M,A,rcond=None) #rxb
    return Vhat[0].T #bxr

def update(U,Y,Vm1,Vp1,lam,tau,gam,ind,iflag,seeds,kappa):
    
    #print(U.shape)

    #print(U.shape)

    UtU = np.dot(U.T,U) # rxr

    PMI_guess = (U @ U.T).copy()

    #print(seeds)

    for c1_idx in range(len(seeds)):
        for c2_idx in range(c1_idx + 1, len(seeds)):
            c1, c2 = seeds[c1_idx], seeds[c2_idx]
            for seed1 in c1:
                for seed2 in c2:
                    PMI_guess[seed1, seed2] = 0
                    PMI_guess[seed2, seed1] = 0

    # similar seeds
    for category in seeds:
        for s1_idx in range(len(category)):
            for s2_idx in range(s1_idx + 1, len(category)):
                seed1, seed2 = category[s1_idx], category[s2_idx]
                #print(seed1, seed2)
                PMI_guess[seed1, seed2] = 1
                PMI_guess[seed2, seed1] = 1
    
    
    r = UtU.shape[0]    
    if iflag:   M   = (1 + kappa * UtU)  + (lam + 2*tau + gam)*np.eye(r)
    else:       M   = (1 + kappa * UtU)  + (lam + tau + gam)*np.eye(r)
    
    Uty = np.dot(U.T,Y) # rxb
    Ub  = U[ind,:].T   # rxb

    #print(Uty.shape, PMI_guess.shape, PMI_guess[ind,:].shape, U.shape, Ub.shape)

    PMI_guess_sub = PMI_guess[ind, :]

    A   = Uty + gam*Ub + tau*(Vm1.T+Vp1.T) + (kappa * PMI_guess @ U).T # rxb
    Vhat = np.linalg.lstsq(M,A,rcond=None) #rxb
    return Vhat[0].T #bxr


#for the above function, the equations are to update V. So:
# Y is n X b (b = batch size)
# r = rank
# U is n X r
# Vm1 and Vp1 are bXr. so they are b rows of V, transposed
import pickle
def import_static_init(T, trainhead):

    with open(f'{trainhead}/static_emb_global.pkl', 'rb') as handle:
        emb = pickle.load(handle)
        
    U, V = copy.deepcopy(emb), copy.deepcopy(emb)
    return U,V

def initvars(vocab_size,T,rank):
    # dictionary will store the variables U and V. tuple (t,i) indexes time t and word index i
    
    U,V = [],[]
    U.append(np.random.randn(vocab_size,rank)/np.sqrt(rank))
    V.append(np.random.randn(vocab_size,rank)/np.sqrt(rank))
    for t in range(1,T):
        U.append(U[0].copy())
        V.append(V[0].copy())
        print(t)
    return U,V
    
import pandas as pd
import scipy.sparse as ss
def getmat(f,v,rowflag):
    data = pd.read_csv(f)

    data = data.values

    X = ss.coo_matrix((data[:,2],(data[:,0],data[:,1])),shape=(v,v))
   
   
    if rowflag: 
        X = ss.csr_matrix(X)
        #X = X[inds,:]
    else:
        X = ss.csc_matrix(X)
        #X = X[:,inds]
    
    return X#.todense()

def getbatches(vocab,b):
    batchinds = []
    current = 0
    while current<vocab:
        inds = range(current,min(current+b,vocab))
        current = min(current+b,vocab)
        batchinds.append(inds)
    return batchinds

#   THE FOLLOWING FUNCTION TAKES A WORD ID AND RETURNS CLOSEST WORDS BY COSINE DISTANCE
from sklearn.metrics.pairwise import cosine_similarity
def getclosest(wid,U):
    C = []
    for t in range(len(U)):
        temp = U[t]
        K = cosine_similarity(temp[wid,:],temp)
        mxinds = np.argsort(-K)
        mxinds = mxinds[0:10]
        C.append(mxinds)
    return C
        
# THE FOLLOWING FUNCTIONS COMPUTES THE REGULARIZER SCORES GIVEN U AND V ENTRIES
def compute_symscore(U,V):
    return np.linalg.norm(U-V)**2

def compute_smoothscore(U,Um1,Up1):
    X = np.linalg.norm(U-Up1)**2 + np.linalg.norm(U-Um1)**2
    return X
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 11:28:53 2022

@author: User
"""

import numpy as np
from functools import reduce
import operator as op
from itertools import  combinations_with_replacement, product
import math
import statistics
from numba import jit, cuda
from scipy.stats import mode
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
import copy
import random
import warnings
import statistics 
warnings.filterwarnings("ignore")

@jit
def bestFitness(pop):  
    best=np.zeros((1,2))
    #for o in range(2):
    for i in range(len(pop)):
        if i==0:
            best=pop[i].bestFitness
        else:
            if pop[i].bestFitness<best:
                best=pop[i].bestFitness
    return best

                
@jit
def NashE(cur_fitness,best):    
    NashE=0       
    for o in range(len(cur_fitness)):    
        NashE+=(cur_fitness[o]-best[o])/best[o]
    return NashE
        
        
@jit        
def find_best_NashE(pop):
    best_NashE=0
    for i in range(len(pop)):
        if i==0:
            best_NashE=pop[i].NE
            j=i
        else:
            if pop[i].NE<best_NashE:
                best_NashE=pop[i].NE
                j=i
    return j




def isEquilibrium(NashE,best_NashE):
    return (NashE - best_NashE<0.5)
        

            


def isDominates(x,y):
    return (x<=y).all() and (x<y).any()

@jit
def determinDomination(pop):
    for i in range(len(pop)):
        pop[i].isDominated=0
    for i in range(0,len(pop)-1):
        for j in range(i+1,len(pop)):
            if isDominates(np.array(pop[i].cur_fitness),np.array(pop[j].cur_fitness)):
                pop[j].isDominated=1
            if isDominates(np.array(pop[j].cur_fitness),np.array(pop[i].cur_fitness)):
                pop[i].isDominated=1

@jit   
def calc_sse(centroids: np.ndarray, labels: np.ndarray, data: np.ndarray):
    distances = 0
    for i, c in enumerate(centroids):
        idx = np.where(labels == i)
        dist = np.sum((data[idx] - c)**2)
        distances += dist
    return distances
       
@jit        
def cdist_fast(XA, XB):

    XA_norm = np.sum(XA**2, axis=1)
    XB_norm = np.sum(XB**2, axis=1)
    XA_XB_T = np.dot(XA, XB.T)
    distances = XA_norm.reshape(-1,1) + XB_norm - 2*XA_XB_T
    return distances



@jit
def medoid(centroids, labels,data):
    med=[]
    for i in range(len(centroids)):
        idx = np.where(labels == i)
        if len(idx[0])!=0 and len(idx[0])!=1 :
            point=data[idx]
            distance=[]
            for p1 in range(len(idx[0])):
                for p2 in range(p1+1,len(idx[0])):
                    distance.append(np.linalg.norm(point[p1] - point[p2])/len(idx))
            med.append(min(distance))
    m=min(med)*len(data)
    return m
            
        

@jit
def clusterConnectivity(f1,centroids, labels,data):
    return f1/medoid(centroids, labels,data)



def Multi_Objectvies(centroids, labels,data):
    
    f1=calc_sse(centroids, labels, data)    
    f2=clusterConnectivity(f1,centroids, labels, data)
    return f1,f2
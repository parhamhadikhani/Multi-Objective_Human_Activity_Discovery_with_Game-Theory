# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 11:33:30 2022

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
from functions import *
warnings.filterwarnings("ignore")

class Archive:
    def __init__(self,partical,size,mesh_div=10):
        self.mesh_div=mesh_div 
        self.particals=partical
        self.size=size
        determinDomination(self.particals)
        tempArchive=[x for x in self.particals if x.isDominated!=1]
        self.archive=copy.deepcopy(tempArchive)
        self.mesh_id=np.zeros(len(self.archive))
        
        self.inflation=0.1
        self.fitness=[par.cur_fitness for par in self.archive] 
        self.f_max=np.max(np.array(self.fitness),axis=0)
        self.f_min=np.min(np.array(self.fitness),axis=0)    
        
    def createGrid(self):
        if len(self.archive)<=1:
            return
        self.f_max=np.max(np.array(self.fitness),axis=0)
        self.f_min=np.min(np.array(self.fitness),axis=0)  
        dc=self.f_max-self.f_min
        self.f_max=self.f_max+self.inflation*dc
        self.f_min=self.f_min-self.inflation*dc 
        self.mesh_id=np.zeros(len(self.archive))   
        for i in range(len(self.archive)):
            self.mesh_id[i]=self._cal_mesh_id(self.fitness[i])
        
    def _cal_mesh_id(self,fit):
        id_=0
        for i in range(len(fit)):
            try:
                id_dim=int((fit[i]-self.f_min[i])*self.mesh_div/(self.f_max[i]-self.f_min[i]))
                id_ = id_ + id_dim*(self.mesh_div**i)
            except ValueError:
                id_dim=0
                id_ = id_ + id_dim*(self.mesh_div**i)
        return id_
    
    def RouletteWheelSelection(self,prob):
        p=np.random.random()
        cunsum=0
        for i in range(len(prob)):
            cunsum+=prob[i]
            if p<=cunsum:
                return i
        
    def selectLeader(self):
    
        if len(self.archive)!=1:
            best=bestFitness(self.archive)
            for j, particle in enumerate(self.archive):
                self.archive[j].NE=NashE(self.archive[j].bestFitness,best)
            gbest= self.archive[find_best_NashE(self.archive)]
            return gbest

        else:
            return self.archive[0]
    
    def deletePartical(self):
        mesh_id=self.mesh_id
        unique_elements,counts_elements = np.unique(mesh_id,return_counts=True)
        p=np.exp(counts_elements)
        p=p/np.sum(p)
        idx=self.RouletteWheelSelection(p)       
        smi=np.where(self.mesh_id==unique_elements[idx])
        smi=list(smi)[0]
        select=np.random.randint(0,len(smi))       
        del self.archive[smi[select]]
        self.mesh_id=np.delete(self.mesh_id,smi[select])
        del self.fitness[smi[select]]
    
    def update(self,particals):
        self.particals=particals
        determinDomination(self.particals)        
        curr_temp=[x for x in self.particals if x.isDominated!=1]
        curr=copy.deepcopy(curr_temp)
        self.archive+=curr
        
        determinDomination(self.archive)
        curr_temp=[x for x in self.archive if x.isDominated!=1]
        self.archive=copy.deepcopy(curr_temp)
        self.fitness=[par.cur_fitness for par in self.archive] 
#___________________________________________________________________________________#
#   A Novel Skeleton-Based Human Activity Discovery
#   Technique Using Particle Swarm Optimization with
#   Gaussian Mutation
#
#                                                                                   #
#   Author and programmer: Parham Hadikhani, DTC Lai, WH Ong                             #
#                                                                                   #
#   e-Mail:20h8561@ubd.edu.bn, daphne.lai@ubd.edu.bn, weehong.ong@ubd.edu.bn        #   
#___________________________________________________________________________________#
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


class Particle:
    
    def __init__(self,
                 n_cluster: int,
                 data: np.ndarray,
                 #nn,
                 wmax: float = 0.9,
                 c1max: float = 2.5,
                 c2max: float = 0,
                 wmin: float = 0.4,
                 c1min: float = 0,
                 c2min: float = 2.5):
        index = np.random.choice(list(range(len(data))), n_cluster)
        self.centroids = data[index].copy()
        self.best_position = self.centroids.copy()
        self.isDominated=0 
        self.label=self._predict(data)
        self.bestFitness = Multi_Objectvies(self.centroids, self.label, data)#best_sse  #nn
        self.cur_fitness=self.bestFitness
        self.velocity = np.zeros_like(self.centroids)
        self._w = wmax
        self._c1 = c1max
        self._c2 = c2min
        self._wmax = wmax
        self._c1max = c1max
        self._c2max = c2max
        self._wmin = wmin
        self._c1min = c1min
        self._c2min = c2min
        self.sigma=1
        self.NE=100
        #self.isEquilibrium=0

    
    def _update_parameters(self,t,tmax):
        self._c1=self._c1max-(self._c1max-self._c1min)*(t/tmax)
        self._c2=(self._c2min-self._c2max)*(t/tmax)+self._c2max
        self._w=self._wmax-(t*(self._wmax-self._wmin))/tmax
        
        if t!=0:
          self.sigma= self.sigma-(1/(tmax))
          
    def _updateFit(self):
        # cur_NE=NashE(self.cur_fitness,self.bestFitness)
        # if isEquilibrium(cur_NE,self.NE):
        #     self.bestFitness=self.cur_fitness
        #     self.best_position=self.centroids.copy()
        if isDominates(np.array(self.cur_fitness),np.array(self.bestFitness)):
        #if self.cur_fitness[0] < self.bestFitness[0]:
            self.bestFitness=self.cur_fitness
            self.best_position=self.centroids.copy()
        elif isDominates(np.array(self.bestFitness),np.array(self.cur_fitness)):
            pass
        else:
            if np.random.random()<0.5:
                self.bestFitness=self.cur_fitness
                self.best_position=self.centroids.copy()
    
    def _update_centroids(self, data: np.ndarray):
        self.centroids = self.centroids + self.velocity
        self.label=self._predict(data)
        self.cur_fitness = Multi_Objectvies(self.centroids, self.label, data )#sss    
    
    def _update_velocity(self, gbest_position: np.ndarray):
        v_old = self._w * self.velocity
        cognitive_component = self._c1 * np.random.random() * (self.best_position - self.centroids)
        social_component = self._c2 * np.random.random() * (gbest_position - self.centroids)
        self.velocity = v_old + cognitive_component + social_component
            
    def _divide_chunks(self,l, n):       
        for i in range(0, len(l), n):  
            yield l[i:i + n]
        
    def _predict(self, data: np.ndarray) -> np.ndarray:
        distance = cdist_fast(data,self.centroids)
        cluster = self._assign_cluster(distance)
        return cluster

    def _assign_cluster(self, distance: np.ndarray) -> np.ndarray:
        cluster = np.argmin(distance, axis=1)
        return cluster
    
    def update(self, gbest_position, data: np.ndarray ,t, tmax):
        self._update_velocity(gbest_position)
        self._update_centroids(data)
        self._updateFit()
        self._update_parameters(t,tmax)
        #self.NE=NashE(self.bestFitness,gbest.bestFitness)

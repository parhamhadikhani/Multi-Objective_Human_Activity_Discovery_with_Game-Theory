# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 08:09:44 2021

@author: Windows
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

from particle import Particle

from archive import Archive

class MOPGMGT:
    def __init__(self,
                 n_cluster: int,
                 n_particles: int,
                 data: np.ndarray,
                 max_iter: int = 50,
                 mesh_div=5):
        self.n_cluster = n_cluster
        self.n_particles = n_particles
        self.data = data
        self.max_iter = max_iter
        self.gbest_centroids = None
        self.mesh_div=mesh_div
        self.archive=[]
        self.BestNash=0
        self.history=[]

        self.particles=[Particle(self.n_cluster, self.data) for i in range(self.n_particles)]
        self.gbest=self.particles[0]

        
    def update_(self,i):
        for j, particle in enumerate(self.particles):
            self.gbest=self.archive.selectLeader()
            self.gbest_centroids=self.gbest.centroids   
            self.history.append(self.gbest.bestFitness)
            particle.update(self.gbest_centroids, self.data, i ,self.max_iter)
        #mutation
        for j, particle in enumerate(self.particles):
            if particle.isDominated!=1:
                for ite in range(10):
                    velocity=particle.velocity*np.exp(np.random.normal(0, particle.sigma))
                    centroids=np.random.normal(0, particle.sigma)*velocity+particle.centroids
                    distance = cdist_fast(self.data,centroids)
                    labelmu = np.argmin(distance, axis=1)
                    current_fitness=Multi_Objectvies(centroids, labelmu,self.data)
                    if isDominates(np.array(current_fitness),np.array(particle.bestFitness)):
                        particle.bestFitness=current_fitness
                        particle.velocity = velocity.copy()
                        particle.centroids = centroids.copy()

        #mutation     
        
        self.archive.update(self.particles) 
            
    def run(self):

        self.archive=Archive(self.particles,self.n_particles)

        for i in range(self.max_iter):
            print(i)
            self.update_(i)
        #print(self.gbest.bestFitness)
        print('Finish with gbest score {:.18f}'.format(self.gbest.bestFitness[0]))
        return  self.history,self.gbest

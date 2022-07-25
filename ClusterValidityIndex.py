# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 12:57:54 2021

@author: Windows
"""

from sklearn.cluster import KMeans
from numba import jit, cuda
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance
from math import sqrt,log2,exp,log
from scipy.special import factorial
#import networkx as nx 
from sklearn.metrics import pairwise_distances
from scipy import spatial
import warnings


warnings.filterwarnings("ignore")

def Hartigan(sse,data):
    #threshold=10
    #H_Rule = threshold+1 
    H_Rule=[]
    for k in range(len(sse)):
        if k!=len(sse)-1:
            H_Rule.append(((float( sse[k])/sse[k+1])-1)*(len(data)-(k+2)-1))
        else:
            H_Rule.append(H_Rule[len(sse)-2]-0.2)
            #return k+2
    #if H_Rule > threshold:
        #k+=1
    return H_Rule

def Krzanowski_Lai(sse,data,features):
    diff=[]
    kl=[]
    for k in range(len(sse)):
        if k==0:
            diff.append((pow((k+2),2/features)*(sse[k])-0.1)-(pow((k+2),2/features)*(sse[k])))
        else:
            diff.append((pow((k+1),2/features)*(sse[k-1]))-(pow((k+2),2/features)*(sse[k])))
    for i in range(len(diff)): 
        if i!=len(diff)-1:
            kl.append(np.abs(diff[i]/diff[i+1]))
        else:
            kl.append(kl[len(diff)-2]-0.2)
    return kl

@jit
def Slope_Statistic(silhouette_score):
    slope=[]
    for k in range(len(silhouette_score)-1):
        slope.append(-(silhouette_score[k+1]-silhouette_score[k])*np.power(silhouette_score[k],0.5))
    slope.append(-(silhouette_score[-1]-.05))
    return slope
def delta_fast(ck, cl, distances):
    values = distances[np.where(ck)][:, np.where(cl)]
    values = values[np.nonzero(values)]

    return np.min(values)
def big_delta_fast(ci, distances):
    values = distances[np.where(ci)][:, np.where(ci)]
    #values = values[np.nonzero(values)]
            
    return np.max(values)

@jit 
def dunn_fast(points, labels):
    """ Dunn index - FAST (using sklearn pairwise euclidean_distance function)
    
    Parameters
    ----------
    points : np.array
        np.array([N, p]) of all points
    labels: np.array
        np.array([N]) labels of all points
    """
    distances = euclidean_distances(points)
    ks = np.sort(np.unique(labels))
    
    deltas = np.ones([len(ks), len(ks)])*1000000
    big_deltas = np.zeros([len(ks), 1])
    
    l_range = list(range(0, len(ks)))
    
    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = delta_fast((labels == ks[k]), (labels == ks[l]), distances)
        
        big_deltas[k] = big_delta_fast((labels == ks[k]), distances)

    di = np.min(deltas)/np.max(big_deltas)
    return di


@jit
def cdist_fast(XA, XB):
    XA_norm = np.sum(XA**2, axis=1)
    XB_norm = np.sum(XB**2, axis=1)
    XA_XB_T = np.dot(XA, XB.T)
    distances = XA_norm.reshape(-1,1) + XB_norm - 2*XA_XB_T
    return distances

def rbfkernel(gamma, distance1):
    return 2*(1-(np.exp(-gamma * distance1)))

def Intercluster(centroids, labels,data):
    intra =0
    for i, c in enumerate(centroids):
        idx = np.where(labels == i)
        temp=data[idx]
        dist=0
        if len(temp)!=0:
            distance1=cdist_fast(temp,temp)
            intra +=np.sum(distance1)/len(temp)
    return intra

def Intracluster(centroids, labels,data):
    Sep=0
    distance1=cdist_fast(centroids,centroids)
    Sep +=np.sum(distance1)
    return Sep

@jit
def CS_index (centroids: np.ndarray, labels: np.ndarray, data: np.ndarray):
	return (Intercluster(centroids, labels,data)/Intracluster(centroids, labels,data))

@jit
def compute_bic(centers,labels,data):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    m=len(centers)
    distances=0
    for i, c in enumerate(centers):
        idx = np.where(labels == i)
        dist =  np.sum((data[idx] - c)**2)
        distances += dist
    N, d = data.shape
    n = np.bincount(labels)
    cl_var = (1.0 / (N - m) / d) *distances
    const_term = 0.5 * m * np.log(N) * (d+1)
    
    BIC = np.sum([n[i] * np.log(n[i]) -n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term
    
    return(np.abs(BIC))

def Distortions(data,centers ):
    distortions = np.repeat(0, len(centers)).astype(np.float32)
    for k in range(len(centers)):
        p=data.shape[1]
        for_mean = np.repeat(0, len(data)).astype(np.float32)
        for i in range(len(data)):
            dists = np.repeat(0, len(centers[k])).astype(np.float32)
            for cluster, c in enumerate(centers[k]):
                tmp = np.transpose(data[i] - c)
                dists[cluster] = tmp.dot(tmp)
            for_mean[i] = min(dists)
        distortions[k] = np.mean(for_mean) / p
    return distortions
        

@jit   
def Jump(data,centroids):
    """ returns a vector of jumps for each cluster """
 
    distortions=Distortions(data,centroids)
    p=data.shape[1]
    Y=p/2
    jumps=np.repeat(0, len(centroids)).astype(np.float32)
    jumps[0]=(distortions[0] ** (-Y))
    for k in range(1, len(distortions)):
        jumps[k]=(distortions[k] ** (-Y) - distortions[k-1] ** (-Y))
    return min(jumps)




#utility

#@jit
def CUN(data_set,clusters):

    U = data_set.shape[0]
    m_l=0
    rez = 0
    for attribute in range(data_set.shape[1]) :
        subsetNum = data_set[:,attribute]
        m_l = np.mean(subsetNum)
        distances = []
        for i in range(len(subsetNum)):
            distances.append((subsetNum[i] - m_l)**2)
        delta_l=np.sum(distances)/U        
        delta_jl_sum = 0        
        for key,cluster in enumerate(clusters):
            a=[]
            for x in cluster:
                if x in subsetNum:
                    a.append(x)
            m_jl = np.mean(a)
            s_l = np.std(a)
            C_j = len(cluster)
            distances = []
            for i in range(len(a)):
                distances.append((a[i] - m_jl)**2)
            delta_jl=np.sum(distances)/C_j
            delta_jl_sum += ((C_j/U) * delta_jl)
            #delta_jl_sum += (C_j/U) * delta_jl
        rez += (delta_l - delta_jl_sum)
    return  rez


#@jit
def Quantization_error_modeling(sse,clusters,data):
    beta=log2(len(clusters))/0.5*log2(sse*len(data))
    pcf=((1/data.shape[1])*sse)*(len(clusters)**-beta)  
    return pcf




@jit
def distance_between_the_datapoints(centroids, labels, data):
    distances =[]
    for i, c in enumerate(centroids):
        if i!=len(centroids):
            idx = np.where(labels == i)
            idx1 = np.where(labels == i+1)
            temp=data[idx]
            temp1=data[idx1]
            dist=0
            for d1 in temp:
                for d2 in temp1:
                    dist+= np.linalg.norm(d1 - d2)
        distances.append(dist)
    return distances
@jit
def similarity_between_the_datapoints(centroids, labels, data):
    distances = 0
    for i, c in enumerate(centroids):
        if i!=len(centroids):
            idx = np.where(labels == i)
            temp=data[idx]
            similiraty=0
            for d1 in range(0,len(temp)-1):
                for d2 in range(1,len(temp)):
                    similiraty+= 1-spatial.distance.cosine(temp[d1] , temp[d2])
        distances += similiraty
    return abs(distances)



def Condorcet_criterion(centroids, labels, data):
    results=np.sum(distance_between_the_datapoints(centroids, labels, data))+similarity_between_the_datapoints(centroids, labels, data)
    return results

@jit
def wcd(centroids, labels, data):
    distances = 0
    for i, c in enumerate(centroids):
        idx = np.where(labels == i)
        dist = np.linalg.norm((data[idx] - c))
        distances += dist
    return distances

#@jit
def bcd(centroids, labels, data):
    distances = 0
    tot=0
    for i, c in enumerate(centroids):
        tot +=c
    tot=tot/len(centroids)
    for i, c in enumerate(centroids):
        idx = np.where(labels == i)
        distances += np.linalg.norm((data[idx] - c))*len(idx)/len(data)*len(centroids)
    return distances
        
def Score_function(centroids, labels, data):
    results=1-1/exp(bcd(centroids, labels, data))+wcd(centroids, labels, data)
    return results
    
@jit    
def get_cop(centroids, labels, data):

    c_intra = wcd(centroids, labels, data)
    c_inter = sorted(distance_between_the_datapoints(centroids, labels, data))[1]

    to_add = len(centroids) * c_intra / c_inter

    return to_add / len(labels)  





@jit
def get_sigma0(data):
		cov=np.cov(np.matrix(data).T)
		sign, logdet =np.linalg.slogdet(cov)
		return logdet
    
@jit
def get_sigma_i(i,labels,data):
    idx = np.where(labels == i)
    si=[]
    for vi in idx:
        si.append(data[vi])
    si=np.array(si)
    si = si.reshape(si.shape[1], (si.shape[0]*si.shape[2]))
    
    cov=np.cov(np.matrix(si).T)
    sign, logdet = np.linalg.slogdet(cov)
    return logdet

@jit
def get_negentropy(centroids, labels, data):
    sum=0
    sum2=0
    for i_clust in range(len(centroids)):
        pi=np.linalg.norm(centroids[i_clust])/np.linalg.norm(centroids[:])
        sum+= pi * get_sigma_i(i_clust,labels,data)
        sum2-= pi * log(pi)
    sum-=get_sigma0(data)
    sum*=0.5
    sum+=sum2
    sum=np.nan_to_num(sum)
    return sum



























# def likelihood(theta, n, x):
#     return (factorial(n) / (factorial(x) * factorial(n - x))) * (theta ** x) * ((1 - theta) ** (n - x))

# def fHPL(clusters, data, n_draws=1000):
#     #prior = pd.Series(sorted(np.random.uniform(0, 1, size=n_draws)))
#     prior_K=len(clusters)/len(data)
#     posterior_probability = likelihood(prior_K,len(data),len(clusters))
#     return posterior_probability















# #without GPU: 0.5514925999996194



# from timeit import default_timer as timer   


# start = timer()
# a=CUN(data,centroids)
# print("with GPU:", timer()-start)


























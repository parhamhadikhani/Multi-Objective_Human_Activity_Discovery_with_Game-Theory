import numpy as np
import statistics
from numba import jit, cuda
#,jitclass
import math 
from scipy.signal import find_peaks
from scipy.stats import mode
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA 


def count_occurrence(list):
    d = {}
    for i in list:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    return d

def purity(groundtruthAssignment, algorithmAssignment):
    purity = 0
    ids = sorted(set(algorithmAssignment)) # sorted unique clusterID
    matching = 0
    for id in ids:
        indices = [i for i, j in enumerate(algorithmAssignment) if j == id]
        cluster = [groundtruthAssignment[i] for i in indices]
        occ = count_occurrence(cluster)
        matching += max(occ.values())
    purity =  matching / float(len(groundtruthAssignment))
    return purity

def cluster_accuracy(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)

def NMI(A,B):
    #sample points
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    # 
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    # 
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat

def F_Score(ind,lbl,label):
    flist=[]
    sum=0
    si=[]
    for i in range(len(ind)):
        if i==0:
            si.append(sum)
        sum+=ind[i]
        si.append(sum)
    for i in range(len(si)):
        if i!=len(si)-1:
            #print('*****activity',i,'*****')
            fscore=0
            for j in range(len(label)):
                fscore+=float(-f1_score(lbl[si[i]:si[i+1]], label[j][si[i]:si[i+1]], average='micro'))
            fscore=float(fscore/len(label))
            flist.append(np.abs(fscore))
            #print('f1_score_mean:',fscore)
        #print('------------------------------\n')
        
    return flist
    

def map1(ind,label):
    n_cluster=len(ind)
    ind=np.array(ind)
    ind2=[]
    ind2.append(0)
    lbl=[]
    for i in range(len(ind)):
        ind2.append(ind2[i]+ind[i])
    for i in range(n_cluster):
        for j in range(ind2[i],ind2[i+1]):
            lbl.append(i)
    lbl=np.array(lbl)
    label1=[]
    wr=np.zeros(len(label))
    contingency_matrix = metrics.cluster.contingency_matrix(lbl, label)
    col_ind=[]
    row_ind=[]
    for i in range(len(contingency_matrix)):
        col_ind.append(np.argmax(contingency_matrix[i,:]))
        row_ind.append(i)
    col_ind=np.array(col_ind)
    row_ind=np.array(row_ind)

    for i in range(len(row_ind)):
        idx = np.where(label == col_ind[i])
        wr[idx] = row_ind[i]
    label1.append(wr)
    return lbl,label1[0]

def conf_mat_draw(mat,a,matcon):
    figsize=(10,10)
    fig, ax = plt.subplots(figsize=figsize)
    #a=['brushing teeth','cooking (chopping)','Rinsing mouth with water','Still(standing)','Taking on the couch','Talking on the phone','Wearing contact lenses','Working on computer','writing on whiteboard','Drinking water','Cooking (stirring)','Opening pill container','Random','Relaxing on couch']
    Sum1=mat.T.sum(axis=1)[:, np.newaxis]
    for i in range(len(Sum1)):
        if(Sum1[i]==0):
            Sum1[i]=1
    sns.heatmap((mat.T/Sum1), square=True, annot=True, fmt='.2f', cbar=False, ax=ax,xticklabels=a, yticklabels=a,linewidths=0.5)
    #sns.heatmap(mat.T/np.sum(mat.T), annot=True, fmt='.2%', cmap='Blues')
    
    e='Predicted label'
    plt.xlabel(e)
    e='True label'
    plt.ylabel(e);
    plt.tight_layout()
    plt.savefig(matcon)
    #plt.show()

#___________________________________________________________________________________#
#   Human Activity Discovery with Automatic Multi-Objective Particle Swarm          #
#   Optimization Clustering with Gaussian Mutation and Game Theory                  #
#                                                                                   #
#                                                                                   #
#   Authors and programmers: Parham Hadikhani, DTC Lai, WH Ong                      #
#                                                                                   #
#   e-Mail:parhamhadikhani@gmail.com, daphne.lai@ubd.edu.bn, weehong.ong@ubd.edu.bn #   
#___________________________________________________________________________________#


import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
import time
from MOPGMGT import *
#from MOPSO_class import *
from Evaluation import *
import csv
from feature_extraction import Extraction
from Preprocessing import *
import warnings
from ClusterValidityIndex import *
from sklearn.metrics import precision_score,recall_score

#dataset=['CAD60','F3D','KARD','UTK','MSR']
dataset=['CAD60']
# Address to reach folder Data and Results 
save_path='{save path }'
save_path=r'C:\Users\User\Downloads\Human-Activity-Discovery-HPGMK-main'

for Datasetsss in dataset:
    

    if Datasetsss=='CAD60':
        maxsub=5
    else:
        maxsub=11
    
    if Datasetsss=='CAD60':
        tag=['brushing teeth','cooking (chopping)','Rinsing mouth with water','Still(standing)','Taking on the couch','Talking on the phone','Wearing contact lenses','Working on computer','writing on whiteboard','Drinking water','Cooking (stirring)','Opening pill container','Random','Relaxing on couch']    
        k=len(tag)
        sample=5
    if Datasetsss=='F3D':
        tag=['wave','drink from a bottle','answer phone','clap','tight lace','sit down','stand up', 'read watch', 'bow']
        k=len(tag)
        sample=5
    if Datasetsss=='KARD':
        tag=['Horizontal arm wave', 	'High arm wave',	'Two hand wave' ,	'Catch Cap' ,	'High throw' ,	'Draw X' ,	'Draw Tick' ,	'Toss Paper' ,	'Forward Kick' ,	'Side Kick' ,	'Take Umbrella' ,	'Bend' ,	'Hand Clap' ,	'Walk' ,	'Phone Call' ,	'Drink' ,	'Sit down' ,	'Stand up']
        k=len(tag)
        sample=5
    if Datasetsss=='UTK':
        tag=['walk', 'sit down', 'stand up', 'pick up', 'carry', 'throw', 'push', 'pull', 'wave hands', 'clap hands']
        k=len(tag)
        sample=5
    if Datasetsss=='MSR':
        tag=['drink', 'eat', 'read book', 'call cellphone', 'write on a paper', 'use laptop', 'use vacuum cleaner', 'cheer up', 'sit still', 'toss paper', 'play game', 'lie down on sofa', 'walk', 'play guitar', 'stand up', 'sit down']
        k=len(tag)
        sample=5
        
    print('Dataset: ****',Datasetsss,'****')
    for subject in range(1,maxsub):
        print('subject: /',subject,'\\')
        data=np.load(r'%s'%save_path+'\Data\%s'%Datasetsss+'\sub%d'%subject+'.npy')
        true_label=np.load(r'%s'%save_path+'\Data\%s'%Datasetsss+'\label%d'%subject+'.npy')
        print('Extracting features: ')
        data,keyframes=Extraction(data)
        data=np.nan_to_num(data)
        #save the results of clustering
        savefile1=r'%s'%save_path+'\Results\\%s'%Datasetsss+'_result_sub_%d'%subject+'.csv'
        savefile2=r'%s'%save_path+'\Results\\%s'%Datasetsss+'_fscore_sub_%d'%subject+'.csv'
        savefile3=r'%s'%save_path+'\Results\\%s'%Datasetsss+'_number_k_sub%d'%subject+'.csv'
        matcon=r'%s'%save_path+'\Results\\%s'%Datasetsss+'_confusion_sub%d'%subject+'.png'
        timesave=r'%s'%save_path+'\Results\\%s'%Datasetsss+'_time_sub%d'%subject+'.npy'
        history1plot=r'%s'%save_path+'\Results\\%s'%Datasetsss+'_history_sub%d'%subject+'.npy'
        
        
        data=dimentaion_reduction(data)
        true_label=Keyframe(true_label,keyframes)
        data,ind=Sampling(data,sample,true_label)
        data = data.reshape(data.shape[0], (data.shape[1]*data.shape[2])) 
        
        _matrix=[]
        _sse=[]
        _accuracy=[]
        _NMI=[]
        _adjusted_rand_score=[]
        _label=[]
        _bestsse=[]
        _time=[]
        _precision_score=[]
        _recall_score=[]
        _v_measure_score=[]
        _fowlkes_mallows_score=[]
        if len(data)>800:#set up maximum number of clusters
            k_max=int(np.sqrt(len(data)))
        else:
            k_max=25
        _obj1=[]
        _obj2=[]
        selection_list=[]
        _estimated_k=[]
        for i in range(1):
            jump=[]
            Best_sse=[]
            for k in range(2,k_max+1):
                start = time.time()
                mopgmgt = MOPGMGT(n_cluster=k_max, n_particles=20,data=data)
                history1,gbest=mopgmgt.run()
                
                selection_list.append(gbest)
                jump.append(Jump(data,gbest.best_position))
                Best_sse.append(gbest.bestFitness[0])
                
            end = time.time()
            runtime=end - start
            _estimated_k.append(np.argmin(jump)+2)
            _bestsse.append(Best_sse[np.argmin(jump)])
            _obj1.append(selection_list[np.argmin(jump)].bestFitness[0])
            _obj2.append(selection_list[np.argmin(jump)].bestFitness[1])
            
            lbl,label=map1(ind,selection_list[np.argmin(jump)].label)
            _label.append(label)
            _matrix.append(confusion_matrix(lbl, label))
            _NMI.append(NMI(lbl, label))
            _accuracy.append(accuracy(lbl, label))

            #_cluster_accuracy.append(cluster_accuracy(lbl, label))
            _adjusted_rand_score.append(adjusted_rand_score(lbl, label))
            _precision_score.append(precision_score(lbl, label, average='micro'))
            _recall_score.append(recall_score(lbl, label, average='micro'))
            _v_measure_score.append(metrics.v_measure_score(lbl, label))
            _fowlkes_mallows_score.append(metrics.fowlkes_mallows_score(lbl, label))
            _time.append(runtime)
            print('number of cluster: ',k,'-----------------','iteration: ',i,'------------------data: %s'%Datasetsss+'----subject: %d'%subject+'---------------------------------------')
            print('purity: ',accuracy(lbl, label))
            
        print('Dataset: ',Datasetsss,'****')
        _time=np.array(_time)
        _sse1=np.array(_bestsse[np.argmax(_accuracy)])
        np.save(timesave,_time)
        np.save(history1plot,_sse1)

        myFile = open(savefile1, 'w')
        with myFile:    
            myFields = ['metric','Mean', 'Min','Max']
            writer = csv.DictWriter(myFile, fieldnames=myFields)    
            writer.writeheader()
            writer.writerow({'metric':'sse','Mean': np.sum(_bestsse)/len(_bestsse), 'Min': np.min(_bestsse),'Max':np.max(_bestsse)})
            writer.writerow({'metric':'Purity','Mean':np.sum(_accuracy)/len(_accuracy),'Min':np.min(_accuracy),'Max':np.max(_accuracy)})
            writer.writerow({'metric':'k ','Mean':round(np.sum(_estimated_k)/30),'Min':_estimated_k[np.argmin(_accuracy)],'Max':_estimated_k[np.argmax(_accuracy)]})
            writer.writerow({'metric':'NMI','Mean':np.sum(_NMI)/len(_NMI),'Min':np.min(_NMI),'Max':np.max(_NMI)})
            writer.writerow({'metric':'precision_score','Mean':np.sum(_precision_score)/len(_precision_score),'Min':np.min(_precision_score),'Max': np.max(_precision_score)})
            writer.writerow({'metric':'recall_score','Mean':np.sum(_recall_score)/len(_recall_score),'Min':np.min(_recall_score),'Max': np.max(_recall_score)})
            writer.writerow({'metric':'pso_v_measure_score','Mean':np.sum(_v_measure_score)/len(_v_measure_score),'Min':np.min(_v_measure_score),'Max': np.max(_v_measure_score)})
            writer.writerow({'metric':'pso_fowlkes_mallows_score','Mean':np.sum(_fowlkes_mallows_score)/len(_fowlkes_mallows_score),'Min':np.min(_fowlkes_mallows_score),'Max': np.max(_fowlkes_mallows_score)})
            writer.writerow({'metric':'adjusted_rand_score','Mean':np.sum(_adjusted_rand_score)/len(_adjusted_rand_score),'Min':np.min(_adjusted_rand_score),'Max':np.max(_adjusted_rand_score)})
        
        flist=F_Score(ind,lbl,_label)
        
        f = open(savefile2, 'w')
        
        with f:
            writer = csv.writer(f)    
            writer.writerow(tag)
            writer.writerow(flist)
        
        file = open(savefile3, 'w')
        
        with file:
            writer = csv.writer(file) 
            writer.writerow(_estimated_k)
            writer.writerow(_estimated_k1)
        
        
        
        conf_mat_draw(_matrix[np.argmax(_accuracy)],tag,matcon)


    









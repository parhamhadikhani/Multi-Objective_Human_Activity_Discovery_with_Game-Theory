# Human Activity Discovery with Automatic Multi-Objective Particle Swarm Optimization Clustering with Gaussian Mutation and Game Theory

Please cite our paper when you use this code in your research.
```
@misc{...,
      title={Human Activity Discovery with Automatic Multi-Objective Particle Swarm Optimization Clustering with Gaussian Mutation and Game Theory}}, 
      author={Parham Hadikhani and Daphne Teck Ching Lai and Wee-Hong Ong},
      year={2022},
      eprint={...},
      archivePrefix={..},
      primaryClass={...}
}
```
## Introduction

This repository contains the implementation of our proposed method for [human activity discovery](https://arxiv.org/abs/2201.05314). Human activity discovery aims to distinguish the activities performed by humans, without any prior information of what defines each activities. 

![arch](/figures/conceptual overview2.jpg)

The workflow of the proposed method is as follows. First, keyframes are selected from the video sequence by computing kinetic energy. Then, features based on different aspects of skeleton including displacement, orientation, and statistical are extracted. Principal components are then chosen by applying PCA on the features. Next, overlapping time windows is used to segment a series of keyframes as activity samples. Hybrid PSO clustering with Gaussian mutation operator is used to discover the groups of activities. Eventually, K-means clustering is applied to the resultant cluster centers to refine the centroids.

![arch](/figures/diagram7.jpg)


### Run
To run the program and get the results, set the save_path address in MAIN.py to reach the Data and Results folder and then run MAIN.py.

### Results
* The average accuracy for all subjects in (a) CAD-60, (b) UTKinect-Action3D, and (c) Florence3D (d) KARD, (e) MSR DailyActivity3D

![arch](/figures/accuracy.png)

* Comparison of confusion matrix of CAD-60 on subject 1.

![arch](/figures/CONF-MAT3.jpg)

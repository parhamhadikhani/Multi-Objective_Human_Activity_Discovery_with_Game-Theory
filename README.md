# Human Activity Discovery with Automatic Multi-Objective Particle Swarm Optimization Clustering with Gaussian Mutation and Game Theory

Please cite our paper when you use this code in your research.
```
@article{hadikhani2023human,
  title={Human activity discovery with automatic multi-objective particle swarm optimization clustering with gaussian mutation and game theory},
  author={Hadikhani, Parham and Lai, Daphne Teck Ching and Ong, Wee-Hong},
  journal={IEEE Transactions on Multimedia},
  year={2023},
  publisher={IEEE}
}
```
## Introduction

This repository contains the implementation of our proposed method for [human activity discovery]([https://ieeexplore.ieee.org/document/10100899]). Human activity discovery aims to distinguish the activities performed by humans, without any prior information of what defines each activities. 

![arch](/figures/conceptual overview2.jpg)

The workflow of the proposed method is as follows. In stage 1, keyframes and joints are and selected from input frames. In stage 2, features are extracted and normalized. Important features are selected b PCA and frames are segmented into the fix sized time overlapping windows. The HAD stage is executed for different values of k in the range of kmin to kmax, and centroids are first selected randomly from samples. After initialization and evaluation of each solution based on the objective functions used, non-dominated solutions are obtained. To avoid getting stuck into the local optimal trap, Gaussian mutation is applied to non-dominated solutions. Then, Nash equilibrium is used to select the global optimal solution. In stage 4, the Jump method is employed to find the optimal number of clusters and return the result of the estimated number of clusters.

![arch](/figures/diagram7.jpg)


### Run
To run the program and get the results, set the save_path address in MAIN.py to reach the Data and Results folder and then run MAIN.py.

### Results
* The average accuracy for all subjects in different datasets

![arch](/figures/accuracy.png)

* Comparison of confusion matrix of different datasets.

![arch](/figures/CONF-MAT3.jpg)

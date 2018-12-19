# Dirichlet-based GP Classification

We provide the code of our paper: \
[Dimitrios Milios, Raffaello Camoriano, Pietro Michiardi,Lorenzo Rosasco, Maurizio Filippone.\ Dirichlet-based Gaussian Processes for Large-scale Calibrated Classification. NeurIPS 2018](http://www.jmlr.org/papers/volume18/16-537/16-537.pdf).


```
@inproceedings{NIPS2018_7840,
	title = {Dirichlet-based Gaussian Processes for Large-scale Calibrated Classification},
	author = {Milios, Dimitrios and Camoriano, Raffaello and Michiardi, Pietro and Rosasco, Lorenzo and Filippone, Maurizio},
	booktitle = {Advances in Neural Information Processing Systems 31},
	editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
	pages = {6008--6018},
	year = {2018},
	publisher = {Curran Associates, Inc.}
}
```



## Prerequisites

GPflow version 1.2.0 or newer:\
https://github.com/GPflow/GPflow/tree/v1.2.0




## Main scripts



### 1. Simple example of Dirichlet-based Classification
**run_dirichletGP_example.py**
is a demonstration of Dirichlet-based classification, where a synthetic dataset is used.



### 2. Evaluation experiments
**run_evaluation_experiment.py** 
performs evaluation of the following methods:
 - gpc: Variational GP classification (Hensman2015)
 - gpd: Our Dirichlet-based approach based on Titsias2009
 - gpr: GPR classification based on Titsias2009
 - gprc: GPR classification with post-hoc calibration (Platt scaling)

We record various metrics (error rate, MNLL, ECE) and the time required for training 
(ie optimisation of hyperparameters and/or variational parameters).

The results can be seen by running the **print_evaluation_result.py** 
script in the _results_ directory.

For the hyperparameter optimisation process we use ScipyOptimizer of gpflow;
optimisation is carried on until convergence is detected.

usage: **run_evaluation_experiment.py DATASET XX**
where:
 - **DATASET** is the directory name of the dataset (see the relevant section)
 - **XX** is the split index (two decimals with a leading zero if needed; eg 01)




### 3. Monitoring experiments
**run_monitoring_experiment.py**
records the progression of various metrics over time for the following methods:
 - gpc: Variational GP classification (Hensman2015)
 - gpc_mb1: Hensman2015 with minibatch of size 1000
 - gpc_mb2: Hensman2015 with minibatch of size 200
 - gpd: Our Dirichlet-based approach based on Titsias2009

The results can be visualised by running the **plot_monitoring_result.py** 
script in the _results_ directory.

For the hyperparameter optimisation process we use AdagradOptimizer of gpflow;
optimisation is carried on for a fixed number of iterations.

usage: **run_evaluation_experiment.py DATASET XX**
where:
 - **DATASET** is the directory name of the dataset (see the relevant section)
 - **XX** is the split index (two decimals with a leading zero if needed; eg 01)






## Datasets

Any datasets should be placed in the **datasets** directory
- bin_classif: subdirectory for binary classification datasets
- mult_classif: subdirectory for multiclass classification datasets


### Required dataset format
The data should have a particular format so that they are compatible with the
**run_evaluation_experiment.py** and **run_monitoring_experiment.py** scripts.

Each _dataset_ directory should contain:
**split** subdirectory with random splits into training and test set:
  - testXX.txt
  - trainXX.txt

In all cases, classes are marked as: 0, 1, 2....
For all datasets, the class is recorded in the **last column** of the corresponding 
testXX.txt and trainXX.txt files.



### Available datasets
A number of datasets can be downloaded and prepared automatically.
Each dataset directory should contain:

  - _split_dataset_ bash script, which creates the contents of the split directory

  - extra readme files and maybe scripts (i.e. download script)

  - the _split_ subdirectory will be ctreated by running the appropriate scripts.

In order to download and prepare all of the available datasets 
run the  **prepare_all_datasets** script in the _datasets_ directory.







## Results

The **results** directory contains the following:

  - _evaluation_: subdirectory for the results of run_evaluation_experiment.py

  - _monitoring_: subdirectory for the results of run_monitoring_experiment.py

  - **print_evaluation_result.py**: 
  Script that prints the results recorded by an evaluation experiment and 
  produces the corresponding reliability plots.

  - **plot_monitoring_result.py**: 
  script that produces a loglog plot of the MNLL progression recorded my a 
  monitoring experiment.




## Source code

The **src** directory contains the following modules:

  - _heteroskedastic.py_:
  Implementation of heteroskedastic GP regression so that it admits 
  a different vector of noise values for each output dimension.

  - _evaluation.py_: code for the evaluation experiments.
  Used by **run_evaluation_experiment.py**

  - _monitoring.py_: code for the evaluation experiments.
  Used by **run_monitoring_experiment.py**

  - _calibration.py_: Implementation of Platt scaling for post-hoc GPR calibration

  - _datasets.py_, _utilities.py_: various utility functions





## References

J. Hensman, A. Matthews, and Z. Ghahramani. 
Scalable Variational Gaussian Process Classification., AISTATS 2015.

M. Titsias. 
Variational Learning of Inducing Variables in Sparse Gaussian Processes, 
AISTATS 2009.


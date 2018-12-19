#!/usr/bin/python3
# Copyright 2018 Dimitrios Milios, Raffaello Camoriano, 
#                Pietro Michiardi,Lorenzo Rosasco, Maurizio Filippone

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tensorflow as tf
import gpflow
gpflow.kernels.RBF(1) # useless code; just to show any warnings early


import os
import sys
import time
import pickle
import random
import numpy as np
from scipy.cluster.vq import kmeans


sys.path.insert(0,'src') # our imports
import datasets
import evaluation
import utilities




## binary datasets: EEG, HTRU2, MAGIC, CovertypeBinary, MiniBoo, SUSY
## multiclass datasets: letter, Drive, MoCap
dataset = ''
split_idx = '01'



## cmd arguments to specify:
## dataset (optional) 
## split_idx (optional)
if len(sys.argv) > 1:
    dataset = str(sys.argv[1])
if len(sys.argv) > 2:
    split_idx = str(sys.argv[2])

if split_idx == '':
    split_idx = '01'
    print('Default split_idx: 01')
if dataset == '':
    print('')
    print('Script that performs evaluation of the following methods:')
    print(' - gpc: Variational GP classification (Hensman2015)')
    print(' - gpd: Our Dirichlet-based approach based on Titsias2009')
    print(' - gpr: GPR classification based on Titsias2009')
    print(' - gprc: GPR classification with post-hoc calibration (Platt scaling)')
    print('')
    utilities.print_usage(os.path.basename(__file__))
    exit()




ARD = False
subset = None
num_inducing = utilities.get_option_inducing(dataset)
a_eps = utilities.get_option_alphaEpsilon(dataset)

use_kmeans = True
test_subset = 20000
GPC_SKIP_THRESHOLD = 1000000
KMEANS_SKIP_THRESHOLD = 500000



path = utilities.get_dataset_path(dataset)
X, y, Xtest, ytest = datasets.load_split(path, split_idx)
X, Xtest = datasets.normalise_oneminusone(X, Xtest)
if subset is not None:
    X = X[:subset, :]
    y = y[:subset]
if Xtest.shape[0] > test_subset:
    Xtest = Xtest[:test_subset, :]
    ytest = ytest[:test_subset]    





report = {}
report['ARD'] = ARD
report['training_size'] = X.shape[0]
report['test_size'] = Xtest.shape[0]
report['num_inducing'] = num_inducing
print('training_size =', X.shape[0])
print('test_size =', Xtest.shape[0])
print('num_inducing =', num_inducing, flush=True)

ytest = ytest.astype(int)
report['ytest'] = ytest

Z = None
if num_inducing is not None:
    if use_kmeans and X.shape[0] <= KMEANS_SKIP_THRESHOLD:
        print('kmeans... ', end='', flush=True)
        start_time = time.time()
        # kmeans returns a tuple
        Z = kmeans(X, num_inducing)[0]
        kmeans_elapsed = time.time() - start_time
        print('done!')
        report['kmeans_elapsed'] = kmeans_elapsed
        print('kmeans_elapsed =', kmeans_elapsed)
    else:
        shuffled = list(range(X.shape[0]))
        random.shuffle(shuffled)
        idx = shuffled[:num_inducing]
        Z = X[idx, :].copy()



'''
Evaluation of the following methods:
 - gpc: Variational GP classification (Hensman2015)
 - gpd: Our Dirichlet-based approach based on Titsias2009
 - gpr: GPR classification based on Titsias2009
 - gprc: GPR classification with post-hoc calibration (Platt scaling)

References:
J. Hensman, A. Matthews, and Z. Ghahramani. 
Scalable Variational Gaussian Process Classification., AISTATS 2015.

M. Titsias. 
Variational Learning of Inducing Variables in Sparse Gaussian Processes, 
AISTATS 2009.
'''


if X.shape[0] <= GPC_SKIP_THRESHOLD:
    tf.reset_default_graph()
    with gpflow.session_manager.get_session().as_default():
        gpc_report = evaluation.evaluate_vgp(X, y, Xtest, ytest, ARD=ARD, Z=Z)
else:
    print('GPC skipped: N >', GPC_SKIP_THRESHOLD)

tf.reset_default_graph()
with gpflow.session_manager.get_session().as_default():
    gpd_report = evaluation.evaluate_gpd(X, y, Xtest, ytest, ARD=ARD, Z=Z, a_eps=a_eps)    

tf.reset_default_graph()
with gpflow.session_manager.get_session().as_default():
    gpr_report = evaluation.evaluate_gpr(X, y, Xtest, ytest, ARD=ARD, Z=Z)

tf.reset_default_graph()
with gpflow.session_manager.get_session().as_default():
    gprc_report = evaluation.evaluate_gpr_calibrated(X, y, Xtest, ytest, ARD=ARD, Z=Z)

report = {**report, **gpc_report, **gpd_report, **gpr_report, **gprc_report}





###############################################################################
# Save results
###############################################################################
save_results = True
result_dir = os.path.join('results', 'evaluation')

if ARD:
    result_path = os.path.join(result_dir, dataset+'_ard_report')
else:
    result_path = os.path.join(result_dir, dataset+'_iso_report')

if save_results:
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_dat = result_path + split_idx + '.dat'
    pickle.dump(report, open(result_dat, 'wb'))

# To load:
# report = pickle.load(open(FILE, "rb" ))


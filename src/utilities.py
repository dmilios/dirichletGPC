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

import os
from pathlib import Path



## Functions that contain experiment settings as used in the paper
## ===============================================================

def get_option_inducing(dataset):
    '''
    Returns the number of inducing points for a given dataset
    as used in the paper.
    '''
    if dataset in ['EEG', 'HTRU2', 'MAGIC', 'SUSY']:
        return 200
    if dataset in ['Drive', 'letter', 'MoCap', 'CovertypeBinary']:
        return 500
    if dataset in ['MiniBoo']:
        return 400
    return None


def get_option_alphaEpsilon(dataset):
    '''
    Returns the alpha_epsilon value for a given dataset
    as used in the paper.
    '''    
    if dataset in ['CovertypeBinary']:
        return 0.1
    if dataset in []:
        return 0.0001
    if dataset in ['letter','Drive','MoCap']:
        return 0.001
    # default:
    # EEG, HTRU2, MAGIC, MiniBoo
    return 0.01





## Misc utility functions 
## ===============================================================

def print_usage(script_name):
    print('')
    print('  usage: ' + script_name + ' DATASET XX\n')
    print('  where:')
    print('   - DATASET is the directory name of a dataset')
    print('   - XX is the split index')
    print('     (two decimals with a leading zero if needed; eg 01)')
    print('')


def get_dataset_path(dataset):
    '''
    Returns the full path of a dataset located in:
    either  datasets/bin_classif  or  datasets/mult_classif
    '''
    path_b = os.path.join("datasets", "bin_classif")
    path_m = os.path.join("datasets", "mult_classif")
    path = os.path.join(path_b, dataset)
    if not Path(path).is_dir():
        path = os.path.join(path_m, dataset)
        if not Path(path).is_dir():
            raise FileNotFoundError('No \'' + dataset + '\' dataset found in: ' 
                + path_b + ' or ' + path_m)
    return path

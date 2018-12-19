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
import numpy as np


def normalise_unitvar (X, Xtest):
    meanX = np.mean(X, 0)
    stdX = np.std(X, 0)
    stdX[stdX == 0] = 1 # to avoid NaN
    X = (X - meanX) / stdX
    Xtest = (Xtest - meanX) / stdX;
    return X, Xtest

def normalise_oneminusone (X, Xtest):
    minx = np.min(X, 0)
    maxx = np.max(X, 0)
    ranges = maxx - minx
    ranges[ranges == 0] = 1 # to avoid NaN
    X = (X - minx) / ranges
    Xtest = (Xtest - minx) / ranges
    X = X * 2 - 1
    Xtest = Xtest * 2 - 1
    return X, Xtest


def load_split(path, split_idx):
    '''
    Assumptions: The 'path' corresponds to a particular dataset
                 which contains the subdirectory 'splits' with contents:
                 train01.txt, test01.txt
                 train02.txt, test02.txt
                 ...
    '''
    if type(split_idx) is not str:
        split_idx = '{:02d}'.format(split_idx)
    path_train = os.path.join(path, 'splits', 'train' + split_idx + '.txt')
    data = np.loadtxt(path_train, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    path_test = os.path.join(path, 'splits', 'test' + split_idx + '.txt')
    data = np.loadtxt(path_test, delimiter=',')
    Xtest = data[:, :-1]
    ytest = data[:, -1]
    return X, y, Xtest, ytest


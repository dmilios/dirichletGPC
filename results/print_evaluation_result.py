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

import os
import sys
import pickle
import numpy as np
import matplotlib.pylab as plt

sys.path.insert(0,'src') # our imports
sys.path.insert(0,'../src') # our imports
import utilities




## binary datasets: EEG, HTRU2, MAGIC, CovertypeBinary, MiniBoo, SUSY
## multiclass datasets: letter, Drive, MoCap
dataset = ''
split_idx = ''


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
    print('Script that prints the results recorded by an evaluation experiment')
    print('and produces the corresponding reliability plots.')
    utilities.print_usage(os.path.basename(__file__))
    exit()


path = os.path.dirname(__file__)
report_filename = dataset + '_iso_report' + split_idx + '.dat'
path = os.path.join(path, 'evaluation', report_filename)

## The result is in the python dictinary named 'report'
report = pickle.load(open(path, "rb" ))


## Separate dictinary entries depending on the key prefix
# - gpc: Variational GP classification (Hensman2015)
# - gpd: Our Dirichlet-based approach based on Titsias2009
# - gpr: GPR classification based on Titsias2009
# - gprc: GPR classification with post-hoc calibration (Platt scaling)
keys = list(report.keys())
keys_gpc = [k for k in keys if k.startswith('gpc_')]
keys_gpd = [k for k in keys if k.startswith('gpd_')]
keys_gpr = [k for k in keys if k.startswith('gpr_')]
keys_gprc = [k for k in keys if k.startswith('gprc_')]
[keys.remove(k) for k in keys_gpc]
[keys.remove(k) for k in keys_gpd]
[keys.remove(k) for k in keys_gpr]
[keys.remove(k) for k in keys_gprc]



def print_scalar_values(keys, report):
    for k in keys:
        v = report[k]
        # three special cases of keys which are ugly to print
        if k.endswith('_prob'):
            print(k + ': ... matrix of predictive probabilities')
        elif k.endswith('_fmu'):
            print(k + ': ... matrix of latent predictive mean')
        elif k.endswith('_fs2'):
            print(k + ': ... matrix of latent predictive variance')
        else:
            print(k + ':', v)


## print the scalar values in the report.
print('')
print('Generic info:')
print('=============')
print_scalar_values(keys, report)

print('')
print('Variational GPC results:')
print('========================')
print_scalar_values(keys_gpc, report)

print('')
print('Dirichlet GPC results:')
print('======================')
print_scalar_values(keys_gpd, report)

print('')
print('GPR results:')
print('============')
print_scalar_values(keys_gpr, report)

print('')
print('GPR (post-hoc calibrated) results:')
print('==================================')
print_scalar_values(keys_gprc, report)



## reliability plot
## ================

# These dicitonaries contain information to build the reliability diagrams
gpc_calib = report['gpc_calib']
gpd_calib = report['gpd_calib']
gpr_calib = report['gpr_calib']
gprc_calib = report['gprc_calib']


def reliability_plot(calibration_report, label='', relative_bin_pos=None):
    '''
    calibration_report: dictinary with the data needed to build the plot
    relative_bin_pos: position of histogram bins relative to other plots
    '''
    conf = calibration_report['conf']
    accu = calibration_report['accu']
    bsizes = calibration_report['bsizes']
    bsizes = bsizes / np.sum(bsizes)
    c = plt.plot(conf, accu, label=label)[0].get_color()
    # Also create histogram that shows the percentage of 
    # predictions in each confidence class
    scale = 0.8
    diff = 0.02 * scale
    width = 0.015 * scale
    bin_positions = conf + diff * relative_bin_pos
    plt.bar(bin_positions, bsizes, width=width,color=c,align='center', alpha=0.4)


xisy = np.linspace(0, 1)
plt.plot(xisy, xisy, 'k--')
reliability_plot(gpc_calib, label='Variational GPC', relative_bin_pos=-1.5)
reliability_plot(gpd_calib, label='Dirichlet GPC', relative_bin_pos=-0.5)
reliability_plot(gpr_calib, label='GPR', relative_bin_pos=0.5)
reliability_plot(gprc_calib, label='GPR (calibrated)', relative_bin_pos=1.5)
plt.xlabel('Confidence')
plt.ylabel('Accuracy')
plt.title(dataset)
plt.legend(loc=2)
plt.show()



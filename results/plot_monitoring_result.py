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


# 
# Script that produces a loglog plot of the MNLL progression 
# recorded my a monitoring experiment.

import os
import sys
import pickle
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl

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
    print('Script that produces a loglog plot of the MNLL progression')
    print('recorded my a monitoring experiment.')
    utilities.print_usage(os.path.basename(__file__))
    exit()




def plot__gpd_vs_gpc(report, metric):
    fig = plt.figure()
    method_labels = {'gpd': 'GPD', 'gpc': 'GPC'}
    metric_labels = {'err': 'Error rate', 'mnll': 'MNLL'}

    method0 = 'gpd' # no batches for method0
    method1 = 'gpc' # lookup batches for method1
    training_size = report['training_size']

    method1_batch_sizes = [training_size]
    all_keys = report.keys()
    method1_keys = list(filter(lambda s: s.startswith(method1), all_keys))
    method1mb_keys = list(filter(lambda s: s.find('mb')>=0, method1_keys))
    for s in method1mb_keys:
        idx = s.find('mb')
        s = s[idx+len('mb'):]
        idx = s.find('_')
        s = s[:idx]
        if not s.isnumeric():
            raise Error('Incompatible report structure')
        s = int(s)
        if s not in method1_batch_sizes:
            method1_batch_sizes.append(s)
    method1_batch_sizes.sort(reverse=True)

    batch_sizes = {}
    batch_sizes[method1] = method1_batch_sizes
    batch_sizes[method0] = [training_size]

    the_colours_to_use = ['C0', 'C3', 'C1', 'C8']
    colours = []
    last_values = []
    cardinalies = []
    for method in [method0, method1]:
        for bsize in batch_sizes[method]:
            prefix = method
            if bsize < training_size:
                prefix += '_' + 'mb' + str(bsize)
            values = report[prefix + '_' + metric + '_values']
            nvalues_full = len(report[method+'_'+metric+'_values'])
            # 'values' is thinned, if much has length larger than nvalues_full
            # so, the function admits both thinned and non-thinned 'values'
            if len(values) > nvalues_full * 1.1:
                iter_per_epoch = training_size / bsize
                values = values[::int(iter_per_epoch)]
            if bsize < training_size:
                bsize_str = ' batch size: ' + str(bsize)
            else:
                bsize_str = ' full'
            label = method_labels[method] + bsize_str
            xx = np.arange(len(values))+1
            line = plt.loglog(xx, values, label=label, linewidth=2,
                color=the_colours_to_use[len(colours)])[0]
            c = line.get_color()
            colours.append(c)
            cardinalies.append(len(values))
            last_values.append(values[-1])

        for i in range(len(cardinalies)):
            colour = colours[i]
            xx = np.arange(cardinalies[i]-1, max(cardinalies))
            fx = last_values[i] * np.ones(len(xx))
            if len(xx) > 0:
                plt.plot(xx, fx, '--', color=colour)

    #plt.yticks(rotation=45)
    axes = fig.get_axes()
    for ax in axes:
        ax.set_yticklabels('')
    plt.ylabel(metric_labels[metric])
    plt.xlabel('Iteration/Epoch')
    y1 = plt.ylim()[1]
    plt.ylim(top=y1*1.2)
    plt.legend()
    plt.tight_layout()



path = os.path.dirname(__file__)
report_filename = dataset + '_iso_report' + split_idx + '.dat'
path = os.path.join(path, 'monitoring', report_filename)
report = pickle.load(open(path, "rb" ))


plot__gpd_vs_gpc(report, 'mnll')
plt.title(dataset)
plt.show()

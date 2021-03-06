#!/usr/bin/python3

import numpy as np
import os



fname = 'MiniBooNE_PID.txt'
data0 = np.loadtxt(fname, skiprows=1)

# The file contains in this order:
# - 36499 signal events (let it be class 0)
# - 93565 background events (let it be class 1)

data = np.zeros((data0.shape[0], data0.shape[1] + 1))
data[:, :data0.shape[1]] = data0
data[36499:, -1] = 1



TEST_SIZE=10000
TRAINING_SIZE = data.shape[0] - TEST_SIZE
SEED = 0


print('dimensions:', data.shape[1] - 1)
print('training size:', TRAINING_SIZE)
print('test size:', TEST_SIZE)
print('classes: ', np.max(data[:, -1]) + 1)
print('seed: ', SEED)


np.random.seed(SEED)
if not os.path.exists('splits'):
    os.makedirs('splits')
for i in range(10):
    rows_idx = np.random.permutation(data.shape[0])
    data_shuffled = data[rows_idx, :]
    data_train = data_shuffled[:TRAINING_SIZE, :]
    data_test = data_shuffled[TRAINING_SIZE:, :]
    
    fname_train = 'splits/train' + '{:02d}'.format(i+1) + '.txt'
    fname_test = 'splits/test' + '{:02d}'.format(i+1) + '.txt'
    np.savetxt(fname_train, data_train, fmt='%.6e', delimiter=',')
    np.savetxt(fname_test, data_test, fmt='%.6e', delimiter=',')
    

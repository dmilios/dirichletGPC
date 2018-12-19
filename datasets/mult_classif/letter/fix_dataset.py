#! /usr/bin/python3

import numpy as np

filename = 'letter-recognition.data'
data = np.loadtxt(filename, delimiter=',')

# moves class column to the end
data = np.roll(data, -1, axis=1)
np.savetxt(filename, data, fmt='%d', delimiter=',')


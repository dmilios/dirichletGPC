#! /usr/bin/python3

import numpy as np

filename = 'Sensorless_drive_diagnosis.txt'
data = np.loadtxt(filename)

# converts classes so that they are in the range [0, 10]
data[:, 48] = data[:, 48] - 1 # was from 1 to 11
np.savetxt(filename, data, fmt='%1.5e', delimiter=',')


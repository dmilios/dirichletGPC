#!/bin/bash

wget -N https://archive.ics.uci.edu/ml/machine-learning-databases/00199/MiniBooNE_PID.txt

python3 split.py

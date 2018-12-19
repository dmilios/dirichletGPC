#!/bin/bash

wget -N https://archive.ics.uci.edu/ml/machine-learning-databases/00391/allUsers.lcl.csv

# replace missing values '?' with '0'
sed -i 's/?/0/g' allUsers.lcl.csv
echo 'Missing values have been replaced with 0'

python3 split.py


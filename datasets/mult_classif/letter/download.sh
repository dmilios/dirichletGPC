#!/bin/bash

wget https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data

sed -i 's/A/0/g' letter-recognition.data # replace A with 0
sed -i 's/B/1/g' letter-recognition.data # replace B with 1
sed -i 's/C/2/g' letter-recognition.data # ...
sed -i 's/D/3/g' letter-recognition.data
sed -i 's/E/4/g' letter-recognition.data
sed -i 's/F/5/g' letter-recognition.data
sed -i 's/G/6/g' letter-recognition.data
sed -i 's/H/7/g' letter-recognition.data
sed -i 's/I/8/g' letter-recognition.data
sed -i 's/J/9/g' letter-recognition.data
sed -i 's/K/10/g' letter-recognition.data
sed -i 's/L/11/g' letter-recognition.data
sed -i 's/M/12/g' letter-recognition.data
sed -i 's/N/13/g' letter-recognition.data
sed -i 's/O/14/g' letter-recognition.data
sed -i 's/P/15/g' letter-recognition.data
sed -i 's/Q/16/g' letter-recognition.data
sed -i 's/R/17/g' letter-recognition.data
sed -i 's/S/18/g' letter-recognition.data
sed -i 's/T/19/g' letter-recognition.data
sed -i 's/U/20/g' letter-recognition.data
sed -i 's/V/21/g' letter-recognition.data
sed -i 's/W/22/g' letter-recognition.data
sed -i 's/X/23/g' letter-recognition.data
sed -i 's/Y/24/g' letter-recognition.data
sed -i 's/Z/25/g' letter-recognition.data

python3 fix_dataset.py


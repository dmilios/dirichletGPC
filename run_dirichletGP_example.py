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


import gpflow
import numpy as np
import scipy.stats
import matplotlib.pylab as plt

import sys
sys.path.insert(0,'src')
import heteroskedastic



N = 20  # training data
np.random.seed(1235)



## create synthetic dataset
## ===============================
xmax = 15
X = np.random.rand(N,1) * xmax
Xtest = np.linspace(0, xmax*1.5, 200).reshape(-1, 1)
Z = X.copy()

y = np.cos(X.flatten()) / 2 + 0.5
y = np.random.rand(y.size) > y
y = y.astype(int)
if np.sum(y==1) == 0:
    y[0] = 1
elif np.sum(y==0) == 0:
    y[0] = 0

# one-hot vector encoding
Y01 = np.zeros((y.size, 2))
Y01[:,0], Y01[:,1] = 1-y, y





## setup heteroskedastic regression
## ================================

# $\alpha_epsilon$ parameter: 
# it can be considered as the parameter of a Dirichlet distribution
# prior to the observation of any label.
a_eps = 0.1

# label transformation
s2_tilde = np.log(1.0/(Y01+a_eps) + 1)
Y_tilde = np.log(Y01+a_eps) - 0.5 * s2_tilde


# For each y, we have two possibilities: 0+alpha and 1+alpha
# Changing alpha (the scale of Gamma) changes the distance
# between different class instances.
# Changing beta (the rate of Gamma) changes the position 
# (i.e. log(alpha)-log(beta)-s2_tilde/2 ) but NOT the distance.
# Thus, we can simply move y for all classes to our convenience (ie zero mean)

# 1st term: guarantees that the prior class probabilities are correct
# 2nd term: just makes the latent processes zero-mean
ymean = np.log(Y01.mean(0)) + np.mean(Y_tilde-np.log(Y01.mean(0)))
Y_tilde = Y_tilde - ymean





## GP setup and hyperparam optimisation
## ====================================

kernel = gpflow.kernels.RBF(1)
model = heteroskedastic.SGPRh(X, Y_tilde, kern=kernel, sn2=s2_tilde, Z=Z)
model.kern.lengthscales = np.std(X)
model.kern.variance = np.var(Y_tilde)


opt = gpflow.train.ScipyOptimizer()
print('\nloglik (before) =', model.compute_log_likelihood())
print('ampl =', model.kern.variance.read_value())
print('leng =', model.kern.lengthscales.read_value())
opt.minimize(model)
print('loglik  (after) =', model.compute_log_likelihood())
print('ampl =', model.kern.variance.read_value())
print('leng =', model.kern.lengthscales.read_value())




## GP prediction
## =============

fmu, fs2 = model.predict_f(Xtest)
fmu = fmu + ymean
lb = fmu - 2 * np.sqrt(fs2)
ub = fmu + 2 * np.sqrt(fs2)

# Estimate mean and quantiles of the Dirichlet distribution through sampling
q=95
mu_dir = np.zeros([fmu.shape[0], 2])
lb_dir = np.zeros([fmu.shape[0], 2])
ub_dir = np.zeros([fmu.shape[0], 2])
source = np.random.randn(1000, 2)
for i in range(fmu.shape[0]):
    samples = source * np.sqrt(fs2[i,:]) + fmu[i,:]
    samples = np.exp(samples) / np.exp(samples).sum(1).reshape(-1, 1)
    Q = np.percentile(samples, [100-q, q], axis=0)
    mu_dir[i,:] = samples.mean(0)
    lb_dir[i,:] = Q[0,:]
    ub_dir[i,:] = Q[1,:]






## Plotting results
## ================

plt.figure(figsize=(12,4))

# to plot the tranformed labels and their standard deviation
Y_tilde += ymean
s1_tilde = np.sqrt(s2_tilde)

plt.subplot(2, 2, 1)
plt.errorbar(X, Y_tilde[:,0], yerr=s1_tilde[:,0], fmt='o', label='Data')
plt.plot(Xtest, fmu[:,0], 'b', label='Posterior')
plt.fill_between(Xtest.flatten(), ub[:,0], lb[:,0], facecolor='0.75')
plt.title('Class 0')
plt.xticks([], [])
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(Xtest, mu_dir[:,0], 'b', label='Prediction')
plt.plot(X, 1-y, 'o', label='Data')
plt.fill_between(Xtest.flatten(), ub_dir[:,0], lb_dir[:,0], facecolor='0.75')
plt.title('Class 0')
plt.xticks([], [])
plt.legend()

plt.subplot(2, 2, 3)
plt.errorbar(X, Y_tilde[:,1], yerr=s1_tilde[:,1], fmt='o', label='Data')
plt.plot(Xtest, fmu[:,1], 'b', label='Posterior')
plt.fill_between(Xtest.flatten(), ub[:,1], lb[:,1], facecolor='0.75')
plt.title('Class 1')
plt.xticks([], [])
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(Xtest, mu_dir[:,1], 'b', label='Prediction')
plt.plot(X, y, 'o', label='Data')
plt.fill_between(Xtest.flatten(), ub_dir[:,1], lb_dir[:,1], facecolor='0.75')
plt.title('Class 1')
plt.xticks([], [])
plt.legend()

plt.show()
quit()

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



'''
This module contains modified code from the sgpr.py module of GPFlow
https://github.com/GPflow/GPflow/blob/develop/gpflow/models/sgpr.py

In particular, the SGPRh class below is a modified version of SGPR
that offers an implementation of heteroskedastic GP regression
so that it admits a different vector of noise values for each output dimension.

This is nessessary for Dirichlet-based GP Classification, as described in the paper:
_Dirichlet-based Gaussian Processes for Large-scale Calibrated Classification_

that appeared in NeurIPS 2018:
https://nips.cc/Conferences/2018/Schedule?showEvent=11583
'''

import numpy as np
import tensorflow as tf

import gpflow
from gpflow.models import GPModel
from gpflow.likelihoods import Likelihood


from gpflow import logdensities ## 'import densities' until GPFlow 1.0.0
from gpflow import settings
from gpflow import transforms
from gpflow.decors import autoflow
from gpflow.decors import params_as_tensors
from gpflow.decors import name_scope
from gpflow.params import DataHolder
from gpflow.params import Parameter
from gpflow.params import Minibatch




class GaussianHeteroskedastic(Likelihood):
    def __init__(self, var=1.0):
        super().__init__()
        if np.isscalar(var):
            var = np.array([var])
        self.variance = Parameter(var, trainable=False)
        self.variance_numel = var.size
        self.variance_ndim = var.ndim

    @params_as_tensors
    def logp(self, F, Y):
        return logdensities.gaussian(F, Y, self.variance)

    @params_as_tensors
    def conditional_mean(self, F):  # pylint: disable=R0201
        return tf.identity(F)

    @params_as_tensors
    def conditional_variance(self, F):
        return tf.fill(tf.shape(F), tf.squeeze(self.variance))

    @params_as_tensors
    def predict_mean_and_var(self, Fmu, Fvar):
        return tf.identity(Fmu), Fvar + self.variance

    @params_as_tensors
    def predict_density(self, Fmu, Fvar, Y):
        return logdensities.gaussian(Fmu, Y, Fvar + self.variance)

    @params_as_tensors
    def variational_expectations(self, Fmu, Fvar, Y):
        return -0.5 * np.log(2 * np.pi) - 0.5 * tf.log(self.variance) \
               - 0.5 * (tf.square(Y - Fmu) + Fvar) / self.variance







class SGPRh(GPModel):
    def __init__(self, X, Y, kern, sn2, Z, mean_function=None, **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        Z is a matrix of pseudo inputs, size M x D
        kern, mean_function are appropriate GPflow objects

        This method only works with a Gaussian likelihood.
        """
        X = DataHolder(X)
        Y = DataHolder(Y)
        likelihood = GaussianHeteroskedastic(sn2)
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)
        self.Z = Parameter(Z)
        self.Z.trainable = False
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]
        self.mean_function = mean_function or gpflow.mean_functions.Zero()

    @params_as_tensors
    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. For a derivation of the terms in here, see the associated
        SGPR notebook.
        """
        num_inducing = tf.shape(self.Z)[0]
        num_data = tf.cast(tf.shape(self.Y)[0], settings.tf_float)
        output_dim = tf.cast(tf.shape(self.Y)[1], settings.tf_float)

        Kdiag = self.kern.Kdiag(self.X)
        Kuf = self.kern.K(self.Z, self.X)
        Kuu = self.kern.K(self.Z) + tf.eye(num_inducing, dtype=settings.tf_float) * settings.jitter
        L = tf.cholesky(Kuu)
        invL_Kuf = tf.matrix_triangular_solve(L, Kuf, lower=True)
        Err = self.Y - self.mean_function(self.X)

        bound = 0
        for i in range(self.num_latent):
            err = tf.slice(Err, [0, i], [self.num_data, 1])
            
            if self.likelihood.variance_ndim > 1:
                sn2 = self.likelihood.variance[:,i]
            else:
                sn2 = self.likelihood.variance
            sigma = tf.sqrt(sn2)

            # Compute intermediate matrices
            # A = inv(L) * Kuf * diag(1/sigma)
            A = invL_Kuf / sigma
            AAT = tf.matmul(A, A, transpose_b=True)
            B = AAT + tf.eye(num_inducing, dtype=settings.tf_float)
            LB = tf.cholesky(B)
            err_sigma = tf.reshape(err, [self.num_data]) / sigma
            err_sigma = tf.reshape(err_sigma, [self.num_data, 1])
            # Aerr = A * (err * diag(1/sigma))
            Aerr = tf.matmul(A, err_sigma)
            c = tf.matrix_triangular_solve(LB, Aerr, lower=True) ## / sigma

            if self.likelihood.variance_numel == 1:
                sum_log_sn2 = num_data * tf.log(sn2)
            else:
                sum_log_sn2 = tf.reduce_sum(tf.log(sn2))
            # compute log marginal bound
            bound += -0.5 * num_data * np.log(2 * np.pi)
            bound += -tf.reduce_sum(tf.log(tf.matrix_diag_part(LB)))
            bound -= 0.5 * sum_log_sn2
            bound += -0.5 * tf.reduce_sum(tf.square(err_sigma))
            bound += 0.5 * tf.reduce_sum(tf.square(c))
            bound += -0.5 * tf.reduce_sum(Kdiag / sn2)
            bound += 0.5 * tf.reduce_sum(tf.matrix_diag_part(AAT))

        return bound


    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. For a derivation of the terms in here, see the associated SGPR
        notebook.
        """
        jitter_level = settings.numerics.jitter_level
        num_inducing = tf.shape(self.Z)[0]
        Kuf = self.kern.K(self.Z, self.X)
        Kuu = self.kern.K(self.Z) + tf.eye(num_inducing, dtype=settings.tf_float) * jitter_level
        Kus = self.kern.K(self.Z, Xnew)
        L = tf.cholesky(Kuu)
        invL_Kuf = tf.matrix_triangular_solve(L, Kuf, lower=True)
        Err = self.Y - self.mean_function(self.X)

        fmean_mat = None
        fvar_mat = None
        for i in range(self.num_latent):
            err = tf.slice(Err, [0, i], [self.num_data, 1])
        
            if self.likelihood.variance_ndim > 1:
                sn2 = self.likelihood.variance[:,i]
            else:
                sn2 = self.likelihood.variance
            sigma = tf.sqrt(sn2)
            # A = inv(L) * Kuf * diag(1/sigma)
            A = invL_Kuf / sigma
            B = tf.matmul(A, A, transpose_b=True) + tf.eye(num_inducing, dtype=settings.tf_float)
            LB = tf.cholesky(B)
            err_sigma = tf.reshape(err, [self.num_data]) / sigma
            err_sigma = tf.reshape(err_sigma, [self.num_data, 1])
            # Aerr = A * (err * diag(1/sigma))
            Aerr = tf.matmul(A, err_sigma)
            c = tf.matrix_triangular_solve(LB, Aerr, lower=True) ## / sigma
            tmp1 = tf.matrix_triangular_solve(L, Kus, lower=True)
            tmp2 = tf.matrix_triangular_solve(LB, tmp1, lower=True)
            mean = tf.matmul(tmp2, c, transpose_a=True)
            if full_cov:
                raise Exception('full_cov not implemented!')
            else:
                var = self.kern.Kdiag(Xnew) + tf.reduce_sum(tf.square(tmp2), 0) \
                    - tf.reduce_sum(tf.square(tmp1), 0)
                shape = tf.stack([1, tf.shape(err)[1]])
                var = tf.tile(tf.expand_dims(var, 1), shape)
                
            if fmean_mat is None or fvar_mat is None:
                fmean_mat = mean
                fvar_mat = var
            else:
                fmean_mat = tf.concat([fmean_mat, mean], 1)
                fvar_mat = tf.concat([fvar_mat, var], 1)

        fmean_mat = fmean_mat + self.mean_function(Xnew)
        return fmean_mat, fvar_mat


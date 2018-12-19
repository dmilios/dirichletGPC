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
import gpflow.training.monitor as mon
from gpflow.models import GPModel

import time
import numpy as np

import heteroskedastic
from evaluation import nll
from evaluation import calibration_test



# if true, preditions will not be saved
SKIP_FULL_RESULTS = True



class LoglikTask(mon.MonitorTask):
    def __init__(self, model: GPModel) -> None:
        super().__init__()
        self.model = model
        self.loglikelihoods = []
    def run(self, context: mon.MonitorContext, *args, **kwargs) -> None:
        if context is not  None:
            ll = context.session.run(self.model._likelihood_tensor)
        else:
            ll = self.model.compute_log_likelihood()
        self.loglikelihoods.append(ll)


class ErrorTask(mon.MonitorTask):
    def __init__(self, model: GPModel, Xtest=None, ytest=None, ymu=0) -> None:
        super().__init__()
        self.model = model
        self.Xtest = Xtest
        self.ytest = ytest.astype(int).flatten()
        self.ymu = ymu
        self.err_values = []
        self.mnll_values = []
        self.elapsed = 0

    def run(self, context: mon.MonitorContext, *args, **kwargs) -> None:
        Xtest = self.Xtest
        ytest = self.ytest
        ymu = self.ymu
        prob = None
        is_gpc = False
        if type(self.model) is gpflow.models.SVGP:
            if type(self.model.likelihood) is gpflow.likelihoods.Bernoulli:
                is_gpc = True
            elif type(self.model.likelihood) is gpflow.likelihoods.MultiClass:
                is_gpc = True
 
        start_time = time.time()
        if is_gpc:
            prob, _ = self.model.predict_y(Xtest)
        else:
            gpd_fmu, gpd_fs2 = self.model.predict_f(Xtest)
            gpd_fmu = gpd_fmu + ymu
            classes = gpd_fmu.shape[1]
            prob = np.zeros(gpd_fmu.shape)
            source = np.random.randn(1000, classes)
            for i in range(gpd_fmu.shape[0]):
                samples = source * np.sqrt(gpd_fs2[i,:]) + gpd_fmu[i,:]
                samples = np.exp(samples) / np.exp(samples).sum(1).reshape(-1, 1)
                prob[i,:] = samples.mean(0)

        if prob.ndim==1 or prob.shape[1]==1:
            pred = prob > 0.5
            pred = pred.astype(int).flatten()
        else:
            pred = np.argmax(prob, 1)
        err_rate = np.mean(pred!=ytest)
        mnll = nll(prob, ytest) / len(ytest)
        self.err_values.append(err_rate)
        self.mnll_values.append(mnll)
        self.elapsed += time.time() - start_time




def optimise(gp_model, step=0.1, maxiter=1000, Xtest=None,ytest=None,ymu=0):
    #opt = gpflow.train.ScipyOptimizer()
    opt = gpflow.train.AdamOptimizer(step) ##see help(tf.train.AdamOptimizer)
    opt = gpflow.train.AdagradOptimizer(step)

    session = gp_model.enquire_session()
    g_step = mon.create_global_step(session)
    errtask = ErrorTask(gp_model, Xtest, ytest, ymu)
    logltask = LoglikTask(gp_model)

    if type(gp_model.X) == gpflow.params.dataholders.Minibatch:
        report_every = int(gp_model.X.shape[0] / gp_model.X._batch_size)
        errtask.with_condition(mon.PeriodicIterationCondition(report_every))
        logltask.with_condition(mon.PeriodicIterationCondition(report_every))

    errtask.run(None) # appends 1st mnll
    logltask.run(None) # appends 1st loglik
    print('\nloglik =', gp_model.compute_log_likelihood())
    with mon.Monitor([errtask, logltask], session, g_step) as m:
        opt.minimize(gp_model,maxiter=maxiter,step_callback=m,global_step=g_step)
    print('\nloglik =', gp_model.compute_log_likelihood())
    return [errtask, logltask]













def monitor_vgp(X, y, Xtest, ytest, ARD=False, Z=None, ampl=None, leng=None,
    minibatch_size=None, maxiter=None):

    report = {}
    dim = X.shape[1]
    if ARD:
        default_len = np.repeat(np.mean(np.std(X,0))*np.sqrt(dim), dim)
    else:
        default_len = np.mean(np.std(X,0))*np.sqrt(dim)
    if Z is None:
        Z=X

    Y = y.reshape(y.size, 1)
    classes = np.max(y).astype(int) + 1
    if classes == 2:
        classes = None # the default value


    rbf_kern = gpflow.kernels.RBF(dim, ARD=ARD, lengthscales=default_len)
    white = gpflow.kernels.White(1, variance=1e-4)
    white.variance.trainable = False
    kernel = rbf_kern + white
    if classes is None:
        lik = gpflow.likelihoods.Bernoulli()
    else:
        lik = gpflow.likelihoods.MultiClass(classes)

    mb_size = minibatch_size
    print('minibatch size :', mb_size)
    method_id = 'gpc'
    if mb_size is not None:
        method_id += '_mb' + str(mb_size)

    gpc_model = gpflow.models.SVGP(X, Y, kern=kernel, likelihood=lik, Z=Z, 
        num_latent=classes, q_diag=False, whiten=True, minibatch_size=mb_size)
    gpc_model.feature.trainable = False


    if ampl is not None:
        rbf_kern.variance.trainable = False
        rbf_kern.variance = ampl * ampl
    if leng is not None:
        rbf_kern.lengthscales.trainable = False
        if ARD:
            rbf_kern.lengthscales = np.ones(dim) * leng
        else:
            rbf_kern.lengthscales = leng


    if maxiter is None:
        maxiter = 100
    if minibatch_size is not None:
        maxiter *= int(X.shape[0] / minibatch_size)

    print(method_id + ' optim... ', end='', flush=True)
    start_time = time.time()    
    [errtask,logltask]=optimise(gpc_model,maxiter=maxiter,Xtest=Xtest,ytest=ytest)
    gpc_elapsed_optim = time.time() - start_time - errtask.elapsed
    print('done!')
    report[method_id + '_elapsed_optim'] = gpc_elapsed_optim
    report[method_id + '_loglikelihoods'] = logltask.loglikelihoods
    report[method_id + '_mnll_values'] = errtask.mnll_values
    report[method_id + '_err_values'] = errtask.err_values


    gpc_amp = np.sqrt(rbf_kern.variance.read_value())
    report[method_id + '_amp'] = gpc_amp
    gpc_len = rbf_kern.lengthscales.read_value()
    report[method_id + '_len'] = gpc_len


    # predict
    print(method_id + ' pred... ', end='', flush=True)
    start_time = time.time()
    gpc_prob, _ = gpc_model.predict_y(Xtest)
    if classes is None:
        gpc_pred = gpc_prob > 0.5
        gpc_pred = gpc_pred.astype(int).flatten()
    else:
        gpc_pred = np.argmax(gpc_prob, 1)
    gpc_elapsed_pred = time.time() - start_time
    print('done!')
    report[method_id + '_minibatch_size'] = mb_size
    report[method_id + '_elapsed_pred'] = gpc_elapsed_pred


    # also get this for reference
    gpc_fmu, gpc_fs2 = gpc_model.predict_f(Xtest)

    if not SKIP_FULL_RESULTS:
        report[method_id + '_pred'] = gpc_pred
        report[method_id + '_prob'] = gpc_prob
        report[method_id + '_fmu'] = gpc_fmu
        report[method_id + '_fs2'] = gpc_fs2


    ytest = ytest.astype(int).flatten()
    gpc_error_rate = np.mean(gpc_pred!=ytest)
    report[method_id + '_error_rate'] = gpc_error_rate

    gpc_ece, conf, accu, bsizes = calibration_test(gpc_prob, ytest)
    report[method_id + '_ece'] = gpc_ece
    gpc_calib = {}
    gpc_calib['conf'] = conf
    gpc_calib['accu'] = accu
    gpc_calib['bsizes'] = bsizes
    report[method_id + '_calib'] = gpc_calib

    gpc_nll = nll(gpc_prob, ytest)
    report[method_id + '_nll'] = gpc_nll

    if classes is None:
        gpc_typeIerror = np.mean(gpc_pred[ytest==0])
        report[method_id + '_typeIerror'] = gpc_typeIerror
        gpc_typeIIerror = np.mean(1-gpc_pred[ytest==1])
        report[method_id + '_typeIIerror'] = gpc_typeIIerror

    print(method_id + '_minibatch_size =', mb_size)
    print(method_id + '_elapsed_optim =', gpc_elapsed_optim)
    print(method_id + '_elapsed_pred =', gpc_elapsed_pred)
    print('---')
    print(method_id + '_amp =', gpc_amp)
    print(method_id + '_len =', gpc_len)
    print('---')
    print(method_id + '_error_rate =', gpc_error_rate)
    if classes is None:
        print(method_id + '_typeIerror =', gpc_typeIerror)
        print(method_id + '_typeIIerror =', gpc_typeIIerror)
    print(method_id + '_ece =', gpc_ece)
    print(method_id + '_nll =', gpc_nll)
    print('\n')
    return report











###############################################################################

def monitor_gpd(X,y,Xtest,ytest,ARD=False,Z=None,ampl=None,leng=None,
    a_eps=0.001,maxiter=None):
    
    report = {}
    dim = X.shape[1]
    if ARD:
        len0 = np.repeat(np.mean(np.std(X,0))*np.sqrt(dim), dim)
    else:
        len0 = np.mean(np.std(X,0))*np.sqrt(dim)
    if Z is None:
        Z=X        

    # prepare y: one-hot encoding
    y_vec = y.astype(int)
    classes = np.max(y_vec).astype(int) + 1
    Y = np.zeros((len(y_vec), classes))
    for i in range(len(y_vec)):
        Y[i, y_vec[i]] = 1


    # label transformation
    s2_tilde = np.log(1.0/(Y+a_eps) + 1)
    Y_tilde = np.log(Y+a_eps) - 0.5 * s2_tilde

    # For each y, we have two possibilities: 0+alpha and 1+alpha
    # Changing alpha (the scale of Gamma) changes the distance
    # between different class instances.
    # Changing beta (the rate of Gamma) changes the position 
    # (i.e. log(alpha)-log(beta)-s2_tilde/2 ) but NOT the distance.
    # Thus, we can simply move y for all classes to our convenience (ie zero mean)

    # 1st term: guarantees that the prior class probabilities are correct
    # 2nd term: just makes the latent processes zero-mean
    ymean = np.log(Y.mean(0)) + np.mean(Y_tilde-np.log(Y.mean(0)))
    Y_tilde = Y_tilde - ymean


    # set up regression
    var0 = np.var(Y_tilde)
    kernel = gpflow.kernels.RBF(dim, ARD=ARD, lengthscales=len0, variance=var0)
    gpd_model = heteroskedastic.SGPRh(X, Y_tilde, kern=kernel, sn2=s2_tilde, Z=Z)
    

    if ampl is not None:
        kernel.variance.trainable = False
        kernel.variance = ampl * ampl
    if leng is not None:
        kernel.lengthscales.trainable = False
        if ARD:
            kernel.lengthscales = np.ones(dim) * leng
        else:
            kernel.lengthscales = leng


    method_id = 'gpd'
    if maxiter is None:
        maxiter = 100

    gpd_elapsed_optim = None
    if ampl is None or leng is None or a_eps is None:
        print(method_id + ' optim... ', end='', flush=True)
        start_time = time.time()
        [errtask, logltask]=optimise(gpd_model,0.1,maxiter,Xtest,ytest,ymean)
        gpd_elapsed_optim = time.time() - start_time - errtask.elapsed
        print('done!')
        report[method_id + '_elapsed_optim'] = gpd_elapsed_optim
        report[method_id + '_loglikelihoods'] = logltask.loglikelihoods
        report[method_id + '_mnll_values'] = errtask.mnll_values
        report[method_id + '_err_values'] = errtask.err_values


    gpd_amp = np.sqrt(gpd_model.kern.variance.read_value())
    report[method_id + '_amp'] = gpd_amp
    gpd_len = gpd_model.kern.lengthscales.read_value()
    report[method_id + '_len'] = gpd_len
    report[method_id + '_a_eps'] = a_eps



    # predict
    print(method_id + ' pred... ', end='', flush=True)
    start_time = time.time()
    gpd_fmu, gpd_fs2 = gpd_model.predict_f(Xtest)
    gpd_fmu = gpd_fmu + ymean

    gpd_prob = np.zeros(gpd_fmu.shape)
    source = np.random.randn(1000, classes)
    for i in range(gpd_fmu.shape[0]):
        samples = source * np.sqrt(gpd_fs2[i,:]) + gpd_fmu[i,:]
        samples = np.exp(samples) / np.exp(samples).sum(1).reshape(-1, 1)
        gpd_prob[i,:] = samples.mean(0)
    
    gpd_elapsed_pred = time.time() - start_time
    print('done!')
    report[method_id + '_elapsed_pred'] = gpd_elapsed_pred




    # the actual prediction
    gpd_pred = np.argmax(gpd_prob, 1)
 
    if not SKIP_FULL_RESULTS:
        report[method_id + '_pred'] = gpd_pred
        report[method_id + '_prob'] = gpd_prob
        report[method_id + '_fmu'] = gpd_fmu
        report[method_id + '_fs2'] = gpd_fs2


    gpd_error_rate = np.mean(gpd_pred!=ytest)
    report[method_id + '_error_rate'] = gpd_error_rate

    gpd_ece, conf, accu, bsizes = calibration_test(gpd_prob, ytest)
    report[method_id + '_ece'] = gpd_ece
    gpd_calib = {}
    gpd_calib['conf'] = conf
    gpd_calib['accu'] = accu
    gpd_calib['bsizes'] = bsizes
    report[method_id + '_calib'] = gpd_calib

    gpd_nll = nll(gpd_prob, ytest)
    report[method_id + '_nll'] = gpd_nll
    if classes == 2:
        gpd_typeIerror = np.mean(gpd_pred[ytest==0])
        report[method_id + '_typeIerror'] = gpd_typeIerror
        gpd_typeIIerror = np.mean(1-gpd_pred[ytest==1])
        report[method_id + '_typeIIerror'] = gpd_typeIIerror


    print(method_id + '_elapsed_optim =', gpd_elapsed_optim)
    print(method_id + '_elapsed_pred =', gpd_elapsed_pred)
    print('---')
    print(method_id + '_amp =', gpd_amp)
    print(method_id + '_len =', gpd_len)
    print(method_id + '_a_eps =', a_eps)
    print('---')
    print(method_id + '_error_rate =', gpd_error_rate)
    if classes == 2:
        print(method_id + '_typeIerror =', gpd_typeIerror)
        print(method_id + '_typeIIerror =', gpd_typeIIerror)    
    print(method_id + '_ece =', gpd_ece)
    print(method_id + '_nll =', gpd_nll)
    print('\n')
    return report


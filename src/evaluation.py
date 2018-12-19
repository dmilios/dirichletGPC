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
gpflow.kernels.RBF(1) # useless code; just to show any warnings early

import time
import numpy as np
from heteroskedastic import SGPRh



################################################################################
### Useful functions
    
def nll(p, y):
    y = y.astype(int).flatten()
    if p.ndim == 1 or p.shape[1] == 1:
        p = p.flatten()
        P = np.zeros([y.size, 2])
        P[:,0], P[:,1] = 1-p, p
        p = P
    classes = p.shape[1]
    Y = np.zeros((y.size, classes))
    for i in range(y.size):
        Y[i,y[i]] = 1
    logp = np.log(p)
    logp[np.isinf(logp)] = -750
    loglik = np.sum(Y * logp, 1)
    return -np.sum(loglik)

def calibration_test(p, y, nbins=10):
    '''
    Returns ece:  Expected Calibration Error
            conf: confindence levels (as many as nbins)
            accu: accuracy for a certain confidence level
                  We are interested in the plot confidence vs accuracy
            bin_sizes: how many points lie within a certain confidence level
    '''
    edges = np.linspace(0, 1, nbins+1)
    accu = np.zeros(nbins)
    conf = np.zeros(nbins)
    bin_sizes = np.zeros(nbins)
    # Multiclass problems are treated by considering the max
    if p.ndim>1 and p.shape[1]!=1:
        pred = np.argmax(p, axis=1)
        p = np.max(p, axis=1)
    else:
        # the treatment for binary classification
        pred = np.ones(p.size)
    #
    y = y.flatten()
    p = p.flatten()
    for i in range(nbins):
        idx_in_bin = (p > edges[i]) & (p <= edges[i+1])
        bin_sizes[i] = max(sum(idx_in_bin), 1)
        accu[i] = np.sum(y[idx_in_bin] == pred[idx_in_bin]) / bin_sizes[i]
        conf[i] = (edges[i+1] + edges[i]) / 2
    ece = np.sum(np.abs(accu - conf) * bin_sizes) / np.sum(bin_sizes)
    return ece, conf, accu, bin_sizes





################################################################################
### Classification with proper VGP

def evaluate_vgp(X, y, Xtest, ytest, ARD=False, Z=None, ampl=None, leng=None):
    report = {}
    dim = X.shape[1]
    if ARD:
        default_len = np.repeat(np.mean(np.std(X,0))*np.sqrt(dim), dim)
    else:
        default_len = np.mean(np.std(X,0))*np.sqrt(dim)

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


    if Z is None:
        gpc_model = gpflow.models.VGP(X, Y, kern=kernel, likelihood=lik, 
            num_latent=classes)
    else:
        gpc_model = gpflow.models.SVGP(X, Y, kern=kernel, likelihood=lik, 
            Z=Z, num_latent=classes, q_diag=False, whiten=True)
        gpc_model.feature.trainable = False
    opt = gpflow.train.ScipyOptimizer()


    if ampl is not None:
        rbf_kern.variance.trainable = False
        rbf_kern.variance = ampl * ampl
    if leng is not None:
        rbf_kern.lengthscales.trainable = False
        if ARD:
            rbf_kern.lengthscales = np.ones(dim) * leng
        else:
            rbf_kern.lengthscales = leng


    print('gpc optim... ', end='', flush=True)
    start_time = time.time()    
    opt.minimize(gpc_model)    
    gpc_elapsed_optim = time.time() - start_time
    print('done!')
    report['gpc_elapsed_optim'] = gpc_elapsed_optim


    gpc_amp = np.sqrt(rbf_kern.variance.read_value())
    report['gpc_amp'] = gpc_amp
    gpc_len = rbf_kern.lengthscales.read_value()
    report['gpc_len'] = gpc_len


    # predict
    print('gpc pred... ', end='', flush=True)
    start_time = time.time()
    gpc_prob, _ = gpc_model.predict_y(Xtest)
    if classes is None:
        gpc_pred = gpc_prob > 0.5
        gpc_pred = gpc_pred.astype(int).flatten()
    else:
        gpc_pred = np.argmax(gpc_prob, 1)
    gpc_elapsed_pred = time.time() - start_time
    print('done!')
    report['gpc_elapsed_pred'] = gpc_elapsed_pred



    # also get this for reference
    gpc_fmu, gpc_fs2 = gpc_model.predict_f(Xtest)

    report['gpc_pred'] = gpc_pred
    report['gpc_prob'] = gpc_prob
    report['gpc_fmu'] = gpc_fmu
    report['gpc_fs2'] = gpc_fs2


    ytest = ytest.astype(int).flatten()
    gpc_error_rate = np.mean(gpc_pred!=ytest)
    report['gpc_error_rate'] = gpc_error_rate

    gpc_ece, conf, accu, bsizes = calibration_test(gpc_prob, ytest)
    report['gpc_ece'] = gpc_ece
    gpc_calib = {}
    gpc_calib['conf'] = conf
    gpc_calib['accu'] = accu
    gpc_calib['bsizes'] = bsizes
    report['gpc_calib'] = gpc_calib

    gpc_nll = nll(gpc_prob, ytest)
    report['gpc_nll'] = gpc_nll

    if classes is None:
        gpc_typeIerror = np.mean(gpc_pred[ytest==0])
        report['gpc_typeIerror'] = gpc_typeIerror
        gpc_typeIIerror = np.mean(1-gpc_pred[ytest==1])
        report['gpc_typeIIerror'] = gpc_typeIIerror

    print('gpc_elapsed_optim =', gpc_elapsed_optim)
    print('gpc_elapsed_pred =', gpc_elapsed_pred)
    print('---')
    print('gpc_amp =', gpc_amp)
    print('gpc_len =', gpc_len)
    print('---')
    print('gpc_error_rate =', gpc_error_rate)
    if classes is None:
        print('gpc_typeIerror =', gpc_typeIerror)
        print('gpc_typeIIerror =', gpc_typeIIerror)
    print('gpc_ece =', gpc_ece)
    print('gpc_nll =', gpc_nll)
    print('\n')
    return report









################################################################################
### Classification with GPR on the transformed Dirichlet distribution

def evaluate_gpd(X,y,Xtest,ytest,ARD=False,Z=None,ampl=None,leng=None,
    a_eps=0.001, scale_sn2=False):

    report = {}
    dim = X.shape[1]
    if ARD:
        len0 = np.repeat(np.mean(np.std(X,0))*np.sqrt(dim), dim)
    else:
        len0 = np.mean(np.std(X,0))*np.sqrt(dim)

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
    if Z is None:
        Z=X
    gpd_model = SGPRh(X, Y_tilde, kern=kernel, sn2=s2_tilde, Z=Z)


    opt = gpflow.train.ScipyOptimizer()
    if ampl is not None:
        kernel.variance.trainable = False
        kernel.variance = ampl * ampl
    if leng is not None:
        kernel.lengthscales.trainable = False
        if ARD:
            kernel.lengthscales = np.ones(dim) * leng
        else:
            kernel.lengthscales = leng

    gpd_elapsed_optim = None
    if ampl is None or leng is None or a_eps is None:
        print('gpd optim... ', end='', flush=True)
        start_time = time.time()
        opt.minimize(gpd_model)
        gpd_elapsed_optim = time.time() - start_time
        print('done!')
        report['gpd_elapsed_optim'] = gpd_elapsed_optim


    gpd_amp = np.sqrt(gpd_model.kern.variance.read_value())
    report['gpd_amp'] = gpd_amp
    gpd_len = gpd_model.kern.lengthscales.read_value()
    report['gpd_len'] = gpd_len
    report['gpd_a_eps'] = a_eps


    # predict
    print('gpd pred... ', end='', flush=True)
    start_time = time.time()
    gpd_fmu, gpd_fs2 = gpd_model.predict_f(Xtest)
    gpd_fmu = gpd_fmu + ymean

    # Estimate mean of the Dirichlet distribution through sampling
    gpd_prob = np.zeros(gpd_fmu.shape)
    source = np.random.randn(1000, classes)
    for i in range(gpd_fmu.shape[0]):
        samples = source * np.sqrt(gpd_fs2[i,:]) + gpd_fmu[i,:]
        samples = np.exp(samples) / np.exp(samples).sum(1).reshape(-1, 1)
        gpd_prob[i,:] = samples.mean(0)

    gpd_elapsed_pred = time.time() - start_time
    print('done!')
    report['gpd_elapsed_pred'] = gpd_elapsed_pred



    # the actual prediction
    gpd_pred = np.argmax(gpd_prob, 1)
 
    report['gpd_pred'] = gpd_pred
    report['gpd_prob'] = gpd_prob
    report['gpd_fmu'] = gpd_fmu
    report['gpd_fs2'] = gpd_fs2


    gpd_error_rate = np.mean(gpd_pred!=ytest)
    report['gpd_error_rate'] = gpd_error_rate

    gpd_ece, conf, accu, bsizes = calibration_test(gpd_prob, ytest)
    report['gpd_ece'] = gpd_ece
    gpd_calib = {}
    gpd_calib['conf'] = conf
    gpd_calib['accu'] = accu
    gpd_calib['bsizes'] = bsizes
    report['gpd_calib'] = gpd_calib

    gpd_nll = nll(gpd_prob, ytest)
    report['gpd_nll'] = gpd_nll
    if classes == 2:
        gpd_typeIerror = np.mean(gpd_pred[ytest==0])
        report['gpd_typeIerror'] = gpd_typeIerror
        gpd_typeIIerror = np.mean(1-gpd_pred[ytest==1])
        report['gpd_typeIIerror'] = gpd_typeIIerror


    print('gpd_elapsed_optim =', gpd_elapsed_optim)
    print('gpd_elapsed_pred =', gpd_elapsed_pred)
    print('---')
    print('gpd_amp =', gpd_amp)
    print('gpd_len =', gpd_len)
    print('gpd_a_eps =', a_eps)
    print('---')
    print('gpd_error_rate =', gpd_error_rate)
    if classes == 2:
        print('gpd_typeIerror =', gpd_typeIerror)
        print('gpd_typeIIerror =', gpd_typeIIerror)    
    print('gpd_ece =', gpd_ece)
    print('gpd_nll =', gpd_nll)
    print('\n')
    return report










################################################################################
### Classification with GPR

def evaluate_gpr(X,y,Xtest,ytest,ARD=False,sigma2=None,Z=None,ampl=None,leng=None):
    report = {}
    dim = X.shape[1]
    if ARD:
        default_len = np.repeat(np.mean(np.std(X,0))*np.sqrt(dim), dim)
    else:
        default_len = np.mean(np.std(X,0))*np.sqrt(dim)

    # prepare y
    Y = y.reshape(y.size, 1)
    classes = np.max(y).astype(int) + 1
    if classes == 2:
        classes = None # the default value
    else:
        y_vec = y.astype(int)
        Y = np.zeros((len(y_vec), classes))
        for i in range(len(y_vec)):
            Y[i, y_vec[i]] = 1
    mean_y = np.mean(Y, 0)
    Y = Y-mean_y

    kernel = gpflow.kernels.RBF(dim, ARD=ARD, lengthscales=default_len)

    if Z is None:
        Z=X
    gpr_model = gpflow.models.SGPR(X, Y, kern=kernel, Z=Z)
    gpr_model.feature.trainable = False
    if sigma2 is not None:
        gpr_model.likelihood.variance = sigma2
        gpr_model.likelihood.variance.trainable = False
    opt = gpflow.train.ScipyOptimizer()


    if ampl is not None:
        kernel.variance.trainable = False
        kernel.variance = ampl * ampl
    if leng is not None:
        kernel.lengthscales.trainable = False
        if ARD:
            kernel.lengthscales = np.ones(dim) * leng
        else:
            kernel.lengthscales = leng


    gpr_elapsed_optim = None
    if ampl is None or leng is None or sigma2 is None:
        print('gpr optim... ', end='', flush=True)
        start_time = time.time()
        opt.minimize(gpr_model)
        gpr_elapsed_optim = time.time() - start_time
        print('done!')
        report['gpr_elapsed_optim'] = gpr_elapsed_optim


    gpr_amp = np.sqrt(gpr_model.kern.variance.read_value())
    report['gpr_amp'] = gpr_amp
    gpr_len = gpr_model.kern.lengthscales.read_value()
    report['gpr_len'] = gpr_len
    gpr_s2 = gpr_model.likelihood.variance.read_value()
    report['gpr_s2'] = gpr_s2


    # predict
    print('gpr pred... ', end='', flush=True)
    start_time = time.time()
    gpr_fmu, gpr_fs2 = gpr_model.predict_f(Xtest)
    gpr_fmu = gpr_fmu + mean_y
    gpr_elapsed_pred = time.time() - start_time
    print('done!')
    report['gpr_elapsed_pred'] = gpr_elapsed_pred

    # simply truncate negatives and normalise
    if classes is None:
        fmu = gpr_fmu.copy()
        fmu[fmu<0] = 0
        fmu[fmu>1] = 1
        gpr_prob = fmu
        # the actual prediction
        gpr_pred = gpr_prob > 0.5
        gpr_pred = gpr_pred.astype(int).flatten()        

    else:
        fmu = gpr_fmu.copy()
        fmu[fmu<0] = 0
        sum_fmu = np.sum(fmu, 1).reshape(-1, 1)
        sum_fmu[sum_fmu==0] = 1
        gpr_prob = fmu / sum_fmu
        # the actual prediction
        gpr_pred = np.argmax(gpr_prob, 1)


    report['gpr_pred'] = gpr_pred
    report['gpr_prob'] = gpr_prob
    report['gpr_fmu'] = gpr_fmu
    report['gpr_fs2'] = gpr_fs2



    gpr_error_rate = np.mean(gpr_pred!=ytest)
    report['gpr_error_rate'] = gpr_error_rate

    gpr_ece, conf, accu, bsizes = calibration_test(gpr_prob, ytest)
    report['gpr_ece'] = gpr_ece
    gpr_calib = {}
    gpr_calib['conf'] = conf
    gpr_calib['accu'] = accu
    gpr_calib['bsizes'] = bsizes
    report['gpr_calib'] = gpr_calib

    gpr_nll = nll(gpr_prob, ytest)
    report['gpr_nll'] = gpr_nll

    if classes is None:
        gpr_typeIerror = np.mean(gpr_pred[ytest==0])
        report['gpr_typeIerror'] = gpr_typeIerror
        gpr_typeIIerror = np.mean(1-gpr_pred[ytest==1])
        report['gpr_typeIIerror'] = gpr_typeIIerror

    print('gpr_elapsed_optim =', gpr_elapsed_optim)
    print('gpr_elapsed_pred =', gpr_elapsed_pred)
    print('---')
    print('gpr_amp =', gpr_amp)
    print('gpr_len =', gpr_len)
    print('gpr_s2 =', gpr_s2)
    print('---')
    print('gpr_error_rate =', gpr_error_rate)
    if classes is None:
        print('gpr_typeIerror =', gpr_typeIerror)
        print('gpr_typeIIerror =', gpr_typeIIerror)        
    print('gpr_ece =', gpr_ece)
    print('gpr_nll =', gpr_nll)
    print('\n')
    return report










################################################################################
### Classification with GPR (post-hoc calibration)

def evaluate_gpr_calibrated(X,y,Xtest,ytest, valid_ratio=0.2, calib_method='platt',
    ARD=False,sigma2=None,Z=None,ampl=None,leng=None):
    '''
    Evaluation of post-hoc calibrated GPR classification
    '''
    if np.max(y).astype(int) == 1:
        return bincf_gpr_calibrated(X=X,y=y,Xtest=Xtest,ytest=ytest, 
            valid_ratio=valid_ratio, calib_method=calib_method, ARD=ARD,
            sigma2=sigma2,Z=Z,ampl=ampl,leng=leng)
    else:
        return mltcf_gpr_calibrated(X=X,y=y,Xtest=Xtest,ytest=ytest, 
            valid_ratio=valid_ratio, calib_method=calib_method, ARD=ARD,
            sigma2=sigma2,Z=Z,ampl=ampl,leng=leng)





def bincf_gpr_calibrated(X,y,Xtest,ytest, valid_ratio=0.2, calib_method='platt',
    ARD=False,sigma2=None,Z=None,ampl=None,leng=None):
    '''
    Binary Classification with GPR
    calib_method: 'platt' | 'isotonic' | None
    '''
    from calibration import ProbitCalibration
    from sklearn.calibration import _SigmoidCalibration
    from sklearn.isotonic import IsotonicRegression

    vsize = int(X.shape[0] * valid_ratio)
    Xvalid = X[:vsize, :]
    yvalid = y[:vsize]
    X = X[vsize:, :]
    y = y[vsize:]
    Xtmp = np.append(Xvalid, Xtest, axis=0)
    ytmp = np.append(yvalid, ytest, axis=0)

    print('Post-hoc calibrated GPR')
    print('Before Calibration -----------------------------')
    report = evaluate_gpr(X,y,Xtmp,ytmp,ARD,sigma2,Z,ampl,leng)

    gpr_pred = report['gpr_pred']
    gpr_prob = report['gpr_prob']
    gpr_fmu = report['gpr_fmu']
    gpr_fs2 = report['gpr_fs2']
    report['gpr_pred'] = gpr_pred[vsize:]
    report['gpr_prob'] = gpr_prob[vsize:]
    report['gpr_fmu'] = gpr_fmu[vsize:]
    report['gpr_fs2'] = gpr_fs2[vsize:]

    # decision function
    # no need to map to [-1,1]; the platt parameters will be adjusted
    df_valid = gpr_fmu[:vsize]
    df_test = gpr_fmu[vsize:]    
    fs2_valid = gpr_fs2[:vsize]
    fs2_test = gpr_fs2[vsize:]
    
    RETRAIN = False
    if RETRAIN:
        X = np.append(Xvalid, X, axis=0)
        y = np.append(yvalid, y, axis=0)
        #ampl, leng, sigma2 = report['gpr_amp'], report['gpr_len'], report['gpr_s2']
        report = evaluate_gpr(X,y,Xtest,ytest,ARD,sigma2,Z,ampl,leng)
        df_test = report['gpr_fmu']
        fs2_test = report['gpr_fs2']

    if calib_method == 'platt':
        df_valid, df_test = df_valid - 0.5, df_test - 0.5
        df_valid = df_valid / np.sqrt(1 + fs2_valid)
        df_test = df_test / np.sqrt(1 + fs2_test)
        calibrator = ProbitCalibration()
        calibrator.fit(df_valid.reshape(-1, 1), yvalid, const_b=True)
        gpr_prob = calibrator.predict(df_test.reshape(-1, 1))
        print('Platt Scaling ----------------------------------')
        report['gpr_platt_A'] = calibrator.a_
        report['gpr_platt_B'] = calibrator.b_
        print('gpr_platt_A =', calibrator.a_)
        print('gpr_platt_B =', calibrator.b_)

    if calib_method == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(df_valid.flatten(), yvalid)
        gpr_prob = calibrator.predict(df_test.flatten())
        print('Isotonic Regression ----------------------------')

    # Just change the report according to Platt scaling or Isotonic regression
    if calib_method == 'platt' or calib_method == 'isotonic':
        # the actual prediction
        gpr_pred = gpr_prob > 0.5
        gpr_pred = gpr_pred.astype(int).flatten()
        report['gpr_pred'] = gpr_pred
        report['gpr_prob'] = gpr_prob

        ytest = ytest.astype(int).flatten()
        gpr_error_rate = np.mean(gpr_pred!=ytest)
        report['gpr_error_rate'] = gpr_error_rate

        gpr_ece, conf, accu, bsizes = calibration_test(gpr_prob, ytest)
        report['gpr_ece'] = gpr_ece
        gpr_calib = {}
        gpr_calib['conf'] = conf
        gpr_calib['accu'] = accu
        gpr_calib['bsizes'] = bsizes
        report['gpr_calib'] = gpr_calib

        gpr_nll = nll(gpr_prob, ytest)
        report['gpr_nll'] = gpr_nll

        gpr_typeIerror = np.mean(gpr_pred[ytest==0])
        report['gpr_typeIerror'] = gpr_typeIerror
        gpr_typeIIerror = np.mean(1-gpr_pred[ytest==1])
        report['gpr_typeIIerror'] = gpr_typeIIerror

        print('gpr_error_rate =', gpr_error_rate)
        print('gpr_typeIerror =', gpr_typeIerror)
        print('gpr_typeIIerror =', gpr_typeIIerror)
        print('gpr_ece =', gpr_ece)
        print('gpr_nll =', gpr_nll)
        print('\n\n')

    report_modified = {}
    for key in report.keys():
        modkey = key[:3] + 'c' + key[3:] # 'gprc_...' instead of 'gpr_...'
        report_modified[modkey] = report[key]
    return report_modified




def mltcf_gpr_calibrated(X,y,Xtest,ytest, valid_ratio=0.2, calib_method='platt',
    ARD=False,sigma2=None,Z=None,ampl=None,leng=None):
    '''
    Multiclass Classification with GPR
    calib_method: 'platt' | 'isotonic' | None

    The classifier is calibrated first for each class separately 
    in an one-vs-all fashion as in scikit-learn:
    http://scikit-learn.org/stable/modules/calibration.html

    Those probabilities do not necessarily sum to one, 
    a postprocessing is performed to normalize them:
    https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/calibration.py#L338
    https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/calibration.py#L379
    '''
    from sklearn.calibration import _SigmoidCalibration
    from sklearn.isotonic import IsotonicRegression

    vsize = int(X.shape[0] * valid_ratio)
    Xvalid = X[:vsize, :]
    yvalid = y[:vsize].astype(int)
    X = X[vsize:, :]
    y = y[vsize:].astype(int)
    Xtmp = np.append(Xvalid, Xtest, axis=0)
    ytmp = np.append(yvalid, ytest, axis=0).astype(int)

    print('Post-hoc calibrated GPR')
    print('Before Calibration -----------------------------')
    report = evaluate_gpr(X,y,Xtmp,ytmp,ARD,sigma2,Z,ampl,leng)

    # fix the report; ignore the validation set
    gpr_pred = report['gpr_pred']
    gpr_prob = report['gpr_prob']
    gpr_fmu = report['gpr_fmu']
    gpr_fs2 = report['gpr_fs2']
    report['gpr_pred'] = gpr_pred[vsize:]
    report['gpr_prob'] = gpr_prob[vsize:, :]
    report['gpr_fmu'] = gpr_fmu[vsize:, :]
    report['gpr_fs2'] = gpr_fs2[vsize:, :]

    # decision function
    # no need to map to [-1,1]; the platt parameters will be adjusted
    df_valid = gpr_fmu[:vsize, :]
    df_test = gpr_fmu[vsize:, :]    
    fs2_valid = gpr_fs2[:vsize, :]
    fs2_test = gpr_fs2[vsize:, :]
    
    RETRAIN = False
    if RETRAIN:
        X = np.append(Xvalid, X, axis=0)
        y = np.append(yvalid, y, axis=0)
        ampl, leng, sigma2 = report['gpr_amp'], report['gpr_len'], report['gpr_s2']
        report = evaluate_gpr(X,y,Xtest,ytest,ARD,sigma2,Z,ampl,leng)
        df_test = report['gpr_fmu']
        fs2_test = report['gpr_fs2']


    classes = df_test.shape[1]
    gpr_prob = np.zeros(df_test.shape)
    Yvalid = np.zeros((len(yvalid), classes))
    for i in range(len(yvalid)):
        Yvalid[i, yvalid[i]] = 1
    gpr_platt_A = np.zeros(classes)
    gpr_platt_B = np.zeros(classes)


    if calib_method == 'platt':
        for c in range(classes):
            calibrator = _SigmoidCalibration()
            calibrator.fit(df_valid[:, c].reshape(-1, 1), Yvalid[:, c])
            gpr_platt_A[c] = calibrator.a_
            gpr_platt_B[c] = calibrator.b_
            gpr_prob[:, c] = calibrator.predict(df_test[:, c].reshape(-1, 1))
        print('Platt Scaling ----------------------------')
        print('gpr_platt_A =', gpr_platt_A)
        print('gpr_platt_B =', gpr_platt_B)
        report['gpr_platt_A'] = gpr_platt_A
        report['gpr_platt_B'] = gpr_platt_B


    if calib_method == 'isotonic':
        for c in range(classes):
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(df_valid[:, c], Yvalid[:, c])
            gpr_prob[:, c] = calibrator.predict(df_test[:, c])
        print('Isotonic Regression ----------------------------')

    # Normalize the probabilities
    gpr_prob /= np.sum(gpr_prob, axis=1)[:, np.newaxis]

    # Just change the report according to Platt scaling or Isotonic regression
    if calib_method is not None:
        # the actual prediction
        gpr_pred = np.argmax(gpr_prob, 1)
        gpr_pred = gpr_pred.astype(int).flatten()
        report['gpr_pred'] = gpr_pred
        report['gpr_prob'] = gpr_prob

        ytest = ytest.astype(int).flatten()
        gpr_error_rate = np.mean(gpr_pred!=ytest)
        report['gpr_error_rate'] = gpr_error_rate

        gpr_ece, conf, accu, bsizes = calibration_test(gpr_prob, ytest)
        report['gpr_ece'] = gpr_ece
        gpr_calib = {}
        gpr_calib['conf'] = conf
        gpr_calib['accu'] = accu
        gpr_calib['bsizes'] = bsizes
        report['gpr_calib'] = gpr_calib

        gpr_nll = nll(gpr_prob, ytest)
        report['gpr_nll'] = gpr_nll

        print('gpr_error_rate =', gpr_error_rate)
        print('gpr_ece =', gpr_ece)
        print('gpr_nll =', gpr_nll)
        print('\n\n')

    report_modified = {}
    for key in report.keys():
        modkey = key[:3] + 'c' + key[3:] # 'gprc_...' instead of 'gpr_...'
        report_modified[modkey] = report[key]
    return report_modified

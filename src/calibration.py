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

from sklearn.calibration import *
from scipy.stats import norm


### Adaptation of _sigmoid_calibration from sklearn.calibration
def _probit_calibration(df, y, sample_weight=None, const_b=True):
    """Probability Calibration with probit method (Platt 2000)
    Parameters
    ----------
    df : ndarray, shape (n_samples,)
        The decision function or predict proba for the samples.
    y : ndarray, shape (n_samples,)
        The targets.
    sample_weight : array-like, shape = [n_samples] or None
        Sample weights. If None, then samples are equally weighted.
    Returns
    -------
    a : float
        The slope.
    b : float
        The intercept.
    References
    ----------
    Platt, "Probabilistic Outputs for Support Vector Machines"
    """
    df = column_or_1d(df)
    y = column_or_1d(y)

    F = df  # F follows Platt's notations
    tiny = np.finfo(np.float).tiny  # to avoid division by 0 warning

    # Bayesian priors (see Platt end of section 2.2)
    prior0 = float(np.sum(y <= 0))
    prior1 = y.shape[0] - prior0
    T = np.zeros(y.shape)
    T[y > 0] = (prior1 + 1.) / (prior1 + 2.)
    T[y <= 0] = 1. / (prior0 + 2.)
    T1 = 1. - T

    def objective(AB):
        # From Platt (beginning of Section 2.2)
        if len(AB) == 1:
            P = norm.cdf(AB[0] * F)
        else:
            P = norm.cdf(AB[0] * F + AB[1])
        l = -(T * np.log(P + tiny) + T1 * np.log(1. - P + tiny))
        if sample_weight is not None:
            return (sample_weight * l).sum()
        else:
            return l.sum()

    if const_b:
        AB0 = np.array([0])
    else:
        AB0 = np.array([0., log((prior0 + 1.) / (prior1 + 1.))])
    AB_ = fmin_bfgs(objective, AB0, disp=False)
    if len(AB_) == 1:
        return AB_[0], 0
    return AB_[0], AB_[1]



### Adaptation of _SigmoidCalibration from sklearn.calibration
class ProbitCalibration(BaseEstimator, RegressorMixin):
    """Probit regression model.
    Attributes
    ----------
    a_ : float
        The slope.
    b_ : float
        The intercept.
    """
    def fit(self, X, y, sample_weight=None, const_b=True):
        """Fit the model using X, y as training data.
        Parameters
        ----------
        X : array-like, shape (n_samples,)
            Training data.
        y : array-like, shape (n_samples,)
            Training target.
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
        Returns
        -------
        self : object
            Returns an instance of self.
        """
        X = column_or_1d(X)
        y = column_or_1d(y)
        X, y = indexable(X, y)

        self.a_,self.b_ = _probit_calibration(X,y,sample_weight,const_b=const_b)
        return self

    def predict(self, T):
        """Predict new data by linear interpolation.
        Parameters
        ----------
        T : array-like, shape (n_samples,)
            Data to predict from.
        Returns
        -------
        T_ : array, shape (n_samples,)
            The predicted data.
        """
        T = column_or_1d(T)
        return norm.cdf(self.a_ * T + self.b_)



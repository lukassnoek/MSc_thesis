# -*- coding: utf-8 -*-
"""
Multivariate encoding: TESTS

Lukas Snoek, Programming: The Next Step
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances as euc_dist
import statsmodels.api as smf

"""
Test 1: Does the manual implementation of multiple regression work
correctly? 
"""

# Some paramters
n_trials = 120
n_features = 5000

# Create predictor
predictor = np.ones((n_trials, n_trials))
for i in xrange(3):
    predictor[i*40:(i*40+40),i*40:(i*40+40)] = 0

X = get_lower_half(predictor)
X = X[:,np.newaxis]

# Create random data and to-be-predicted random RDM
random_data = np.random.normal(0, 5, size = (n_trials, n_features))
random_RDM = euc_dist(random_data)    
y = get_lower_half(random_RDM)
y = y[:,np.newaxis]
y = y - np.mean(y) # demeaning to avoid fitting intercept
    
# Use regression from statsmodels package
model = smf.OLS(y,X)
results = model.fit() 
results.params
results.tvalues

# Check against manual implementation
coeff = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)),X.T), y)
y_hat = np.dot(coeff, X.T) # y-predicted
    
MSE = np.mean((y - y_hat)**2)
var_est = MSE * np.diag(np.linalg.pinv(np.dot(X.T,X)))
SE_est = np.sqrt(var_est)    
t_vals = coeff / SE_est

print "Statsmodels t-value: %f, manual impl. t-value: %f" % (results.tvalues,
                                                             t_vals)

"""
Decision: although the manual implementation seems to be right, it might
be better to implement the statsmodels function, because it's WAY faster.
"""

"""
Test 2: Does regressing a distance matrix introduce a positive bias, i.e.
does it return relatively higher t-values than expected? To test this, we'll
do an regression onto random data to see whether it returns an inflated
number of false positives 
"""

iterations = 1000
t_vals = np.zeros(iterations)
p_vals = np.zeros(iterations)

for i in xrange(iterations):
    random_data = np.random.normal(0, 5, size = (n_trials, n_features))
    random_RDM = euc_dist(random_data)    
    y = get_lower_half(random_RDM)
    y = y[:,np.newaxis]
    y = y - np.mean(y) # demeaning to avoid fitting intercept
    
    model = smf.OLS(y,X)
    results = model.fit() 
    t_vals[i] = float(results.tvalues)
    p_vals[i] = float(results.pvalues)
    print str(i)

n_falsepos = np.sum(p_vals < 0.05) / float(iterations)

print "Over %i iterations, nr. of false positives: %s" % (iterations, n_falsepos)

"""
Decision: It rather seems that the analysis is more conservative than liberal/
biased, which I cannot really explain. I guess any covariance that exists 
within the distances of the trials from the same class is present in distances
of trials across classes, which averages out. So I think that regressing
values from a distance matrix does NOT introduce a bias.
"""
 
# -*- coding: utf-8 -*-
"""
Multivariate encoding: TESTS

Lukas Snoek, Programming: The Next Step
"""
home_dir = '/media/lukas/Data/Matlab2Python/'
script_dir = '/home/lukas/Dropbox/ResMas_UvA/Thesis/Git/Analysis_scripts/modules/'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances as euc_dist
import statsmodels.api as smf

import sys
sys.path.append(script_dir)
import MultivariateEncoding as MultEnc

"""
Test 1: Does the manual implementation of multiple regression work
correctly? 
"""
n_trials = 120
n_features = 1000
predictor = np.ones((n_trials, n_trials))
for i in xrange(3):
    predictor[i*40:(i*40+40),i*40:(i*40+40)] = 0

X = MultEnc.get_lower_half(predictor)
X = X[:,np.newaxis]

# Create random data and to-be-predicted random RDM
random_data = np.random.normal(0, 5, size = (n_trials, n_features))
random_RDM = euc_dist(random_data)    
y = MultEnc.get_lower_half(random_RDM)
y = y[:,np.newaxis]
y = y - np.mean(y) # demeaning to avoid fitting intercept
    
# Use regression from statsmodels package
model = smf.OLS(y,X)
results = model.fit()
results.tvalues
results.pvalues

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

iterations = 2000
t_vals = np.zeros(iterations)
p_vals = np.zeros(iterations)

for i in xrange(iterations):
    random_data = np.random.normal(0, 3, size = (n_trials, n_features))
    random_RDM = euc_dist(random_data)    
    y = MultEnc.get_lower_half(random_RDM)
    y = y[:,np.newaxis]
    y = y - np.mean(y) # demeaning to avoid fitting intercept
    
    model = smf.OLS(y,X)
    results = model.fit() 
    t_vals[i] = float(results.tvalues)
    p_vals[i] = float(results.pvalues)
    print str(i)

n_falsepos = np.sum(p_vals < 0.05) / float(iterations)

f, axarr = plt.subplots(1,2)
axarr[0].hist(t_vals, 15)
axarr[0].set_title('T-value distribution')
axarr[0].set_xlabel('Count')
axarr[0].set_ylabel('T-value')

axarr[1].hist(p_vals, 15)
axarr[1].set_title('P-value distribution')
axarr[1].set_xlabel('Count')
axarr[1].set_ylabel('P-value')

f.tight_layout()
f.savefig(os.path.join(os.getcwd(),'tdist_test.png'))

print "Over %i iterations, nr. of false positives: %s" % (iterations, n_falsepos)

"""
Decision: It rather seems that the analysis is more conservative than liberal/
biased, which I cannot really explain. I guess any covariance that exists 
within the distances of the trials from the same class is present in distances
of trials across classes, which averages out. So I think that regressing
values from a distance matrix does NOT introduce a bias.
"""

""" Test 3: Does the t-value from the regression of the observed RDM scale
nicely with the amount of noise added to simulated data?"""


simulated = np.ones((n_trials, n_trials))
for i in xrange(3):
    simulated[i*40:(i*40+40),i*40:(i*40+40)] = 0


noise = np.arange(1,15,0.5)
t_val = np.zeros(noise.size)
p_val = np.zeros(noise.size)

for i,ns in enumerate(noise):
    
    holder = np.zeros((iterations, 2))
    for j in xrange(iterations):
        sim_plus_noise = simulated + np.random.normal(0,ns,(predictor.shape))

        y = MultEnc.get_lower_half(sim_plus_noise - np.mean(sim_plus_noise))
        model = smf.OLS(y,X)
        results = model.fit() 
        holder[j,0] = float(results.tvalues)
        holder[j,1] = float(results.pvalues)
    
    t_val[i] = np.mean(holder[:,0])
    p_val[i] = np.mean(holder[:,1])
    
plt.plot(t_val,linewidth= 3)
plt.ylabel('T-value')
plt.xlabel('Noise added (in std of normal dist)')
plt.title('T-value as a function of noise added to true effect')
plt.xticks(range(0,t_val.shape[0]), noise)



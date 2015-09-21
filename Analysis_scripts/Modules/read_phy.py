# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 20:34:22 2015

@author: lukas
"""

import numpy as np
import matplotlib.pyplot as plt

test = np.loadtxt(open('/media/lukas/Data/Sample_fMRI/ToProcess/HWW_008_s1_zinnen1.phy'))

n = 0
while True:
    result = np.sum(test[n,6:-1])    
    
    if result > 0:
        break
    
    n += 1

for cPlot in xrange(test.shape[1]):
    plt.subplot(2,5,cPlot+1)
    plt.plot(test[n:n+100,cPlot])
    plt.tight_layout()
    plt.savefig('test.pdf')

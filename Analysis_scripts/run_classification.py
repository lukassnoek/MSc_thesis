# -*- coding: utf-8 -*-
"""
Script to run main classification analysis.
Dynamic Affect project. 
"""

################################ SETUP dirs ################################
import sys
import os
home = os.path.expanduser("~")
script_dir = os.path.join(home, 'LOCAL', 'Analysis_scripts')
sys.path.append(script_dir)    

################################ SETUP params ################################
from modules.main_classify import mvpa_classify
feat_dir = os.path.join(home,'DecodingEmotions')
os.chdir(feat_dir)

# Parameters
identifier = 'merged'
iterations = 30
n_test = 2
zval = 2.3
method = 'fstat'
mvpa_classify(identifier, iterations, n_test, zval, method)
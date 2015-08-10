# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:43:03 2015

@author: lsnoek1
"""

# Set-up
from Modules import main_classify
from os.path import expanduser
home = expanduser("~")
import os

import sys
script_dir = os.path.join(home,'Dropbox','ResMas_UvA','Thesis','Git','Analysis_scripts')
sys.path.append(script_dir)

if sys.platform[0:5] == 'linux':
    data_dir = os.path.join('/run/user/1000/gvfs',
    'smb-share:server=bigbrother.fmg.uva.nl,share=fmri_projects$',
    'fMRI Project DynamicAffect')
else:
    data_dir = os.path.join('Z:/','fMRI Projects','fMRI Project DynamicAffect')

feat_dir = os.path.join(data_dir,'Firstlevel_posttest')
os.chdir(feat_dir)

# Parameters
iterations = 20
n_test = 2
zval = 2.5
main_classify.mvpa_classify(iterations, n_test, zval)
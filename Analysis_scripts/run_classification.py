# -*- coding: utf-8 -*-
"""
Script to run main classification analysis.
Dynamic Affect project. 
"""

################################ SETUP dirs ################################
import sys
import platform
from os.path import expanduser
home = expanduser("~")
import os

if 'Windows' in platform.platform():
    data_dir = os.path.join('Z:/','fMRI Projects','fMRI Project DynamicAffect')
    script_dir = os.path.join(home,'Dropbox','ResMas_UvA','Thesis','Git',
                              'Analysis_scripts')
    
if 'precise' in platform.platform():
    data_dir = os.path.join(home,'.gvfs','storage on bigbrother')
    script_dir = os.path.join(home,'Git','Analysis_scripts')

if 'vivid' in platform.platform():
    data_dir = os.path.join('/run/user/1000/gvfs',
                            'smb-share:server=bigbrother.fmg.uva.nl,share=fmri_projects$',
                            'fMRI Project DynamicAffect')
    script_dir = os.path.join(home,'Dropbox','ResMas_UvA','Thesis','Git',
                              'Analysis_scripts')    

sys.path.append(script_dir)    

################################ SETUP params ################################
from Modules.main_classify import mvpa_classify
feat_dir = os.path.join(data_dir,'DecodingEmotions')
os.chdir(feat_dir)

# Parameters
identifier = 'merged'
iterations = 30
n_test = 4
zval = 2.3
method = 'fstat'
main_classify.mvpa_classify(identifier,iterations, n_test, zval, method)
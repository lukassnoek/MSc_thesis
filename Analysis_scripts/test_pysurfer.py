# -*- coding: utf-8 -*-
"""
Created on Sat May 16 17:33:31 2015

@author: lukas
"""
import os
from surfer import Brain, io

os.environ['SUBJECTS_DIR'] = "/usr/local/freesurfer/subjects"
os.environ['FREESURFER_HOME'] = "/usr/local/freesurfer"
os.environ['MNI_DIR'] = "/usr/local/freesurfer/mni"

brain = Brain('fsaverage', 'split', 'inflated', views=['lat', 'med'])
volume_file = '/home/lukas/cope1_mni.nii.gz'
reg_file = os.path.join(os.environ["FREESURFER_HOME"],
                        "average/mni152.register.dat")
zstat = io.project_volume_data(volume_file, "lh", reg_file)

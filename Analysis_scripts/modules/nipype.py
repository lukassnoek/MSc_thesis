# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:12:14 2015

@author: lukas
"""

# Importing necessary packages
import nipype.interfaces.fsl as fsl
import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util
from IPython.display import Image
import os
import glob
import warnings
import shutil

import sys
sys.path.append("/home/lukas/Dropbox/ResMas_UvA/Thesis/Git/Analysis_scripts/modules")

# Specification of pipeline-general variables
homeDir = '/media/lukas/Data/Sample_fMRI' 
ToProcess = homeDir + '/ToProcess'
subject_stem = 'HWW_'
study_stem = 'PM'

import createDirStructure as cds
os.chdir(homeDir)

cds.convert_parrec2nifti(remove_nifti = 1, backup = 1)
sub_dirs = cds.setup_analysis_skeleton(homeDir,'HWW')
cds.movefiles_subjectdirs(sub_dirs, ToProcess)

infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),
name="infosource")
infosource.iterables = ('subject_id', ['HWW_007','HWW_008'])

datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                                               outfields=['func', 'struct']),
                                               name = 'datasource')
datasource.inputs.base_directory = ToProcess
datasource.inputs.template = '%s/%s/%s.nii'




T1_paths,T1_names = cds.get_filepaths('WIP_sT1',ToProcess)

skullstrip = pe.Node(interface=fsl.BET(), name = 'skullstrip')
skullstrip.iterables= ('frac',[0.6, 0.7])
skullstrip.inputs.in_file = T1_paths[1]
skullstrip.inputs.out_file = T1_paths[1] + "_test"
bet_results = skullstrip.run()
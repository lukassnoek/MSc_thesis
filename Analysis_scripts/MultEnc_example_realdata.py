# -*- coding: utf-8 -*-
#################### EXAMPLE ANALYSIS ########################################
""" The multivariate encoding analysis can be done on a sample
dataset from the Decoding Emotions project (Oosterwijk, Snoek, & Scholte, in
preparation). For more background on the theory/set-up, see: 
https://db.tt/lnAg8n6I. Open the file "MultEnc_example_realdata.py" for
an example how the analysis works on a real dataset (this dataset has only
one factor with three levels though). The file "MultEnc_example_simdata.py" 
contains an example on simulated data with two factors (+ interaction term).
"""

##############################################################################

import os
import sys

# Setting some parameters to let Python know where we're working from (home_dir)
# and where the MultivariateEncoding module can be found (script_dir)
# NB: fill in the absolute paths!
home_dir = '/media/lukas/Data/Matlab2Python/'
script_dir = '/home/lukas/Dropbox/ResMas_UvA/Thesis/Git/Analysis_scripts/modules/'

# We'll append the location of the module to PYTHONPATH
import sys
sys.path.append(script_dir)

# And we'll load in the functions/classes of MultivariateEncoding
import MultivariateEncoding as MultEnc

# Let's go to the home_dir and declare some variables used during the analysis!
os.chdir(home_dir) 
subject_stem = 'HWW' # included in the name of the subject directories

# Paths to the FSL masks and to the MNI 152 (2mm) nifti file, which is
# used as a common anatomical template.
mask_dir = os.path.join(home_dir,'FSL_masks/ROIs/Lateralized_masks')
MNI_path = os.path.join(home_dir,'FSL_masks/MNI152_T1_2mm_brain.nii.gz')

# We're going to use the Harvard-Oxford Probabilistic Cortical Atlas. 
# Because it is probabilistic, we don't want everything > 0 from the mask,
# as it would include a lot outside the brain. We'll set the threshold at 30.
mask_threshold = 30

# Let's pick a first Region of Interest (ROI) to perform our analysis on.
mask = os.path.join(mask_dir, 'InsularCortex_left.nii.gz') # left insula

# We're going to work mostly from the FirstLevel directory
FirstLevel_dir = os.path.join(home_dir, 'FirstLevel')
os.chdir(FirstLevel_dir)

verbose = 1 # we want to see some printed output

MultEnc.create_subject_mats(mask,subject_stem,mask_threshold,verbose)
MultEnc.merge_runs(verbose)



#t_map, scores = main(subject_stem, mask_dir, mask_threshold, MNI_path, prop_train, z_thres)

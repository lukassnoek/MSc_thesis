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
import matplotlib.pyplot as plt

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
mask = os.path.join(mask_dir, 'TemporalPole_left.nii.gz') # left temporal pole

# We're going to work mostly from the FirstLevel directory
FirstLevel_dir = os.path.join(home_dir, 'FirstLevel')
os.chdir(FirstLevel_dir)

# we want to see some printed output; in the main() function, this option
# is turned out by default, as it would clutter the terminal output.
verbose = 1 

# The first step we need to perform, is to create the subject mvpa-matrices
MultEnc.create_subject_mats(mask,subject_stem,mask_threshold,verbose)

# The just created subject-specific mvpa_mat objects are stored in the 
# directory "/mvpa_mats" as a cPickle file (Python's way to store data other
# than simple numpy arrays)

# Now we need to merge the runs; it's a necessary but unintersting step.
MultEnc.merge_runs(verbose)

# Let's check out a mvpa_mat object! 
subject_x = cPickle.load(open(FirstLevel_dir + '/mvpa_mats/HWW_004_merged.cPickle'))

# Check out the attributes of this mvpa_mat instance
attrs = vars(subject_x)
print ', \n \n'.join("%s: %s" % item for item in attrs.items())

# Of interest here are the classes of the trials of subject_x. This dataset
# has 120 trials, divided over three classes (action, interoception, situation).
# Trial 1:40 = action
# Trial 41:80 = interoception
# Trial 81:120 = situation
subject_x.class_labels

""" The hypothesis of this multivariate encoding analysis is, basically, that
IF a certain ROI encodes information about a stimulus category, the trials
within a category (or "class") should be more similar to each other than
trials from different categories (classes). That's why we'll create an RDM,
representational dissimilarity matrix, a fancy word for a pairwise distance
matrix of trials in our design, with the function create_RDM(). 
Don't pay attention to the prop_train and z_thres arguments. These were 
implemented for a feature selection/cross-validation option, but couldn't 
be finished in time. 
"""
RDM,test_idx = MultEnc.create_RDM(subject_x, prop_train = 1, z_thres = None)

# The RDM looks like this:
plt.imshow(RDM,origin='lower')
plt.colorbar(label = 'Euclidian distance')
plt.xlabel('Trials')
plt.ylabel('Trials')    
plt.title('RDM of %s' % subject_x.mask_name)
plt.show()

# We can formalize our hypothesis (i.e. trials across classes are more dis-
# similar than trials within classes) by creating a "predictor RDM", in which
# trials within classes have a predicted distance of 0, and between classes 
# a distance of 1.   
predictor_RDM = MultEnc.create_regressors(subject_x, test_idx, plot = 1)

# Let's check whether the observed RDM contains any information about 
# the classes of our trials.
t_val, p_val = MultEnc.test_RDM(observed = RDM, predictors = predictor_RDM)

print "Regression of mask %s yielded a t-value of %f" % (subject_x.mask_name, t_val),
print "with a corresponding p-value of %f" % p_val 

# Seems that this region doesn't encode class information. Note that in the
# main() analysis, the regression is performed on subject-average RDM, which
# has more power/SNR!

""" What we have done so far is a simplified version of the analysis for 
one mask, for one subject. The entire analysis uses data from all participants
and loops over different gray-matter masks (ROIs). To run the entire analysis,
call the function main(), which integrates all functions in one large
loop over masks. It also nicely backprojects significant t-values onto an
MNI template (it also saves the corresponding nifti file). Evaluate the 
following line to run it (note: this may take about 30 minutes!)."""
 
MultEnc.main(subject_stem, mask_dir, mask_threshold, MNI_path, prop_train, 
         z_thres, verbose = 0)


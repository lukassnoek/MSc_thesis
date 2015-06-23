# -*- coding: utf-8 -*-
#################### MULTIVARIATE ENCODING ###################################
""" Multivariate Encoding module, created for the Research Master course
Programming: The Next Step, at the University of Amsterdam by Lukas Snoek.
This module contains the source code to run a multivariate encoding analysis
on single trial fMRI data. All code necessary to correctly load in single
trial data, perform minor pre-analysis processing steps (normalization of beta
coefficients), and conduct the final multivariate encoding analysis, is 
included in this module. This module is divided into four main parts.

1. Importing packages
Here, necessary packages (all open-source) are imported at the top-most level

2. The main analysis function, main()
This main analysis encompasses the entire analysis, including pre-analysis
processing. It returns a (FDR) thresholded t-value map and plots accordingly.

3. Pre-analysis processing
Here, all pre-analysis tools/functions are defined. The most important function
is create_subject_mats(), which loops over subjects to import single trial
data (as raw beta weights, which are normalized by their SE in the same 
function), extract useful information, and package it all in a user-defined 
class 'mvpa_mat', which is saved (for each subject separately) in the directory
/mvpa_mats. The function merge_runs() merges data from the two separate runs
per subject and saves it again as an mvpa_mat object.
N.B.: These scripts (and the module in general) assume that 'regular' fMRI
preprocessing has been performed for optimal results (incl. motion and slice
time correction, high-pass filtering, and spatial smoothing'). It furthermore
assumes, critically, that all data is transformed to MNI152 (2mm) space.

4. Analysis functions
This part contains the core analysis functions. create_RDM() computes 
a pairwise distance matrix given an mvpa_mat instance. create_regressors()
computes regressor RDMs under the assumption that if trials belong to the
same class, their distance is 0; when they are of different classes, their
distance is 1. It uses the information in mvpa_mat.class_labels to create 
these regressor RDMs. It currently supports both single factor and 2 factor
designs with any amount of levels each. test_RDM() takes as input the observed
RDM (from create_RDM) and the regressors (from create_regressors) and regresses
observed onto the predictors. It returns the t-values associated with the 
coefficient(s) of the factor(s).
The RDM and the corresponding t-value(s) are initialized as an RDM() object
(in the main analysis).

Dependencies (all open source):
- numpy
- scikit-learn 
- matplotlib
- nibabel
- nipy (for 2D plotting)
- statsmodels

It furthermore depends on single-trial data as processed by FSL's GLM
toolbox 'FEAT' (all open-source): http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FEAT.
(ref: M. Jenkinson, C.F. Beckmann, T.E. Behrens, M.W. Woolrich, S.M. Smith. 
FSL. NeuroImage, 62:782-90, 2012).

This module was created and tested on a Ubuntu 15.04 platform with the
Spyder 2.7 IDE (using Python version 2.7). It has not been tested on Windows/
OS X. There could be some problems with path separators (/ vs. \).

Lukas Snoek, spring 2015
"""
##############################################################################

_author_ = "Lukas Snoek"

#################### 1. IMPORTING PACKAGES ###################################

import os
import cPickle
import numpy as np
import glob
import nibabel as nib
import csv
import cPickle
from sklearn import preprocessing as preproc
from nipy.labs.viz import plot_map
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances as euc_dist
import itertools
import statsmodels.sandbox.stats.multicomp as sm
import statsmodels.api as smf

#################### 2. MAIN ANALYSIS ########################################
    
def main(subject_stem, mask_dir, mask_threshold, MNI_path, prop_train, 
         z_thres, verbose = 0):
    """ This function performs the main encoding analysis by looping over
    different masks (region of interests, ROIs). For each mask, it creates
    an mvpa_mat (with .data of trials x features from mask) for each subject.
    It subsequently averages de RDMs over subjects and performs the regression
    analyses and stores the t-value corresponding to the mask.
    Important: This function assumes that it is executed in the FirstLevel
    directoy!
    
    Args:
        subject_stem:       study-specific subject-prefix (here: 'HWW')
        mask_dir:           directory in which FSL masks are stored
        mask_threshold:     minimum for probabilistic FSL masks
        MNI_path:           path to MNI152 nifti file
        prop_train:         proportion of training trials, if cross-validated
                            feature selection is performed (if not: prop_train
                            = 1)
        z_thres:            If prop_train < 1, z_thres refers to the z-value
                            cutoff scores for the difference index during
                            univariate feature selection.
        verbose:            if 1, subfunctions print output (default = 0)
    
    Returns:
        t_map:              ndarray of size MNI.shape (91, 109, 91) with 
                            t-values for each evaluated mask inserted.
        scores:             list of tuples with mask name [0] and corresponding
                            t-value [1]       
    """
    
    # Create RDM directory if it doesn't exist yet, and subfolder for
    # specific mask-list (e.g. bilateral, lateralized, other MNI masks)
    RDM_dir = os.getcwd() + '/RDMs'
    if not os.path.exists(RDM_dir):
        os.makedirs(RDM_dir)
    
    RDM_mask_dir = os.getcwd() + '/RDMs/' + os.path.basename(mask_dir)
    if not os.path.exists(RDM_mask_dir):
        os.makedirs(RDM_mask_dir)

    # Get absolute paths of masks (with .nii.gz extension)
    mask_paths = glob.glob(mask_dir + '/*nii.gz')
    
    # Load MNI nifti file, extract affine
    MNI = nib.load(MNI_path) 
    affine = MNI.get_affine()
    MNI_dims = MNI.get_shape()
    
    # Initialize empty t-value map, corresponding list, and mask-indices
    t_map = np.zeros((MNI.get_shape()))
    t_vals = np.zeros((len(mask_paths)))
    p_vals = np.zeros((len(mask_paths)))
    
    # This variable will store the indices of each mask
    mask_indices = np.zeros((len(mask_paths), MNI_dims[0], MNI_dims[1],MNI_dims[2]))
    
    # This was a vector to store permutation statistics, but I decided to 
    # use standard significance testing instead, because permutation-statistics
    # appeared to be extremely liberal.
    #perm_vec = np.zeros((len(mask_paths), permutations))
    
    # Start loop over masks (ROIs)
    for j,mask in enumerate(mask_paths):
        
        # Create subject matrices with specific maks
        create_subject_mats(mask,subject_stem,mask_threshold,verbose,norm_method = 'univariate')
        merge_runs(verbose)
        
        # Define subject paths to pickle files
        rel_paths = glob.glob('mvpa_mats/*' + subject_stem + '*merged.cPickle')
        sub_paths = [os.path.abspath(path) for path in rel_paths]
        
        # This is somewhat ugly, but there is no way to know how many trials
        # there are without loading a subject file first 
        n_trials = 120 * prop_train
        
        # Initialization of RDM (subjects x RDM row x RDM col)
        RDM_holder = np.zeros((len(sub_paths), n_trials, n_trials))    
    
        # Loop over subjects to load in data
        for i,sub in enumerate(sub_paths):
            mvpa_data = cPickle.load(open(sub))
            RDM_holder[i,:,:],test_idx = create_RDM(mvpa_data, prop_train, z_thres)
         
        # Average RDMs, create regressor, run regression
        av_RDM = np.mean(RDM_holder, axis = 0)
        Predictors = create_regressors(mvpa_data,test_idx, plot = 0)
        t_vals[j], p_vals[j] = test_RDM(av_RDM, Predictors)
        
        # Old permutation code (not used anymore)        
        #perm_vec[j,:] = permute_RDM(avRDM, Reg, permutations)
        #p_vals[j] = (np.sum(perm_vec[j,:] > float(t_vals[j]))) / float(permutations)
        
        # Fill t-map with t-value for that mask
        mni_idx = mvpa_data.mask_index.reshape(MNI.get_shape())
        t_map[mni_idx] = t_vals[j]
        
        # Save mask-indices
        mask_indices[j,mni_idx] = True  
        
        # Save RDM object (reuse attributes from last mvpa_mat instance)
        class_lab = mvpa_data.class_labels
        mask_name = mvpa_data.mask_name
        RDM_to_save = RDM(av_RDM, t_vals[j], mask_name, mni_idx, class_lab)        
        
        with open(RDM_mask_dir + '/' + mask_name[:-7] + '.cPickle', 'wb') as handle:
            cPickle.dump(RDM_to_save, handle)
        
        # Print progress and t-value
        print "Mask: %s," % os.path.basename(mask),
        print 't-value = %f,' % t_vals[j],
        print 'p-value = %f' % p_vals[j],
        print '(' + str(j+1) + '/' + str(len(mask_paths)) + ') '
        
    # Zip scores(mask, corresponding t-value)
    scores = zip([os.path.basename(mask) for mask in mask_paths], t_vals)    
    
    # FDR correction!
    fdr_corr = sm.multipletests(p_vals, method = 'fdr_bh')
    
    # Set insignificant masks to zero
    for i,bool_mask in enumerate(fdr_corr[0]):
        if not bool_mask:
            t_map[mask_indices[i,:,:,:].astype(bool)] = 0    
    
    # Plot FDR corrected t-value map
    thres_map = np.ma.masked_less(t_map, 0.1)
    title = ''
    plot_map(thres_map, affine, draw_cross = False, title = title, 
             annotate = False)
    plt.savefig(os.path.join(os.getcwd(), 'plot_tmap.png'))
    
    # Save a nifti file of the t-map to view in e.g. FSLview
    img = nib.Nifti1Image(t_map, np.eye(4))
    nii_name = os.path.basename(mask_dir) + '_tmap'
    nib.save(img, nii_name)
    
    return(t_map, scores)
    
##############################################################################

#################### 3. PRE-ANALYSIS PROCESSING ##############################
""" These functions (and classes) take care of the 'preprocessing' of single
trial fMRI data as created by the software package FSL. Preprocessing includes
parsing the design.con file to extract class names (extract_class_vector),
loading in the single trial data & normalization by SE, initialization of
mvpa_mat object and saving it as a cPickle file. It also includes the function
merge_runs(), which is only applicable to this specific dataset, that merges
trial data split across two runs. 

Note: All functions should be executed within the FirstLevel directory 
(as created by default by FSL). 
"""

class mvpa_mat():
    """ Object mvpa_mat, consisting of single trial data (trials x features) and
    accompanying information, such as mask (which voxels are used) and labels
    (to which classes do the trials belong). Inherits implicitly from 'object'.
    
    Atrributes:
        data:          single-trial voxel patterns (trials x features)
        subject_name:  name of subject to which the data belongs
        mask_name:     the filename of the mask from which features are indexed
        mask_shape:    shape of mask (usually MNI: 91 x 109 x 91)
        class_labels:  labels to indicate class for each trial
        
    """
    def __init__(self, data, subject_name, mask_name, mask_index, class_labels):
        """ Initializes mvpa_mat object. Uses mainly arguments given with
        the call to the class. 

        Remaining attributes:
            n_features:    number of voxels
            n_trials:      number of trials IN TOTAL
            class_names:   unique class names
            n_class:       length(class_names)
            n_inst:        number of trials PER CLASS              
            class_idx:     indices of trials within classes
            num_labels:    numeric equivalent to class_labels (useful for
                           classification analyses with sklearn)
                          
        """        
        # Initialized attributes        
        self.data = data
        self.subject_name = subject_name
        self.mask_name = mask_name
        self.mask_index = mask_index        
        self.class_labels = class_labels        
        
        # Computed attributes (based on initialized attributes)
        self.n_features = self.data.shape[1]
        self.n_trials = self.data.shape[0]
        
        class_names = []
        [class_names.append(i) for i in class_labels if not class_names.count(i)]
        self.class_names = class_names
        self.n_class = len(class_names)        
        self.n_inst = self.n_trials / self.n_class
        self.class_idx = [np.arange(self.n_inst*i,self.n_inst*i+self.n_inst) \
                          for i in range(self.n_class)]
                            

def extract_class_vector(subject_directory):
    """ Extracts class of each trial and returns a vector of class labels,
    by parsing the design.con file that is produced during the FSL first
    level analysis (FEAT). Is called within the create_subject_mats() function.
    Raises OSError if there is no design.con file.
    """
    
    sub_name = os.path.basename(os.path.normpath(subject_directory))
    to_parse = subject_directory + '/design.con'
    
    # Read in design.con
    if os.path.isfile(to_parse):
        with open(to_parse, 'r') as con_file:
            con_read = csv.reader(con_file, delimiter='\t')
            class_labels = []            
            
            # Extract class of trials until /NumWaves
            for line in con_read:     
                if line[0] == '/NumWaves':
                    break
                else:
                    class_labels.append(line[1])
        
        class_labels = [s.split('_')[0] for s in class_labels]
        return(class_labels)
                    
    else:
        raise OSError('There is no design.con file for ' + sub_name)
    
def create_subject_mats(mask,subject_stem,mask_threshold,
                        verbose, norm_method = 'univariate' ):
    """ Creates subject-specific MVPA matrices. It is designed to 
    work with single trial voxel patterns created in a first level analyses 
    by the software package FSL: M. Jenkinson, C.F. Beckmann, T.E. Behrens, 
    M.W. Woolrich, S.M. Smith. FSL. NeuroImage, 62:782-90, 2012.
    
    More specifically, it does the following for each subject:
    1. Load mask, creates index for single trials patterns (actually not in 
       subject loop)
    2. Load in trials, index with mask
    3. Normalizes the patterns (raw beta weights) with the betas SE (sqrt(var)).
    4. Scales the data (zero mean, unit variance)
    5. Initializes mvpa_mat object with data, class_labels, and mask info
    6. Saves mvpa_mat object as pickle file in mvpa_mats directory
    
    Args: 
        mask:             mask to index constrasts, either 'fstat' or a specific 
                          ROI. The exact filename should be given 
                          (e.g. 'graymatter.nii.gz').
        subject_stem:     project-specific subject-prefix
        mask_threshold:   minimum threshold of probabilistic FSL mask
        norm_method:      normalization method, either 'univariate' (default)
                          or 'multivariate'
        verbose:          if 1, information about the output of function is 
                          printed.
                          
    Returns:
    Nothing, but creates a dir ('mvpa_mats') with individual pickle files.  
    
    Raises:
        ValueError:       If the nr. of paths to statistic files do not match
                          the number of trials (from class_labels) OR when
                          the nr. of paths is zero.
    """
    
    subject_dirs = glob.glob(os.getcwd() + '/*/*' + subject_stem + '*.feat')
    mat_dir = os.getcwd() + '/mvpa_mats'
    
    if not os.path.exists(mat_dir):
        os.makedirs(mat_dir)
    
    # Load mask, create index, define number of features (voxels)
    mask_name = os.path.basename(mask)
    mask_vol = nib.load(mask)
    mask_index = mask_vol.get_data().ravel() > mask_threshold
    n_features = np.sum(mask_index)    
    
    # Loop over subject directories
    for sub_path in subject_dirs:
        
        # Extract class vector (see function definition below)
        class_labels = extract_class_vector(sub_path)
        
        sub_name = os.path.basename(sub_path).split(".")[0]
        
        if verbose:
            print "Processing %s ... " % sub_name,
        
        # Generate and sort paths to stat files (FSL COPE files)
        stat_paths = glob.glob(sub_path + '/stats_new/cope*mni.nii.gz')
        stat_paths = sort_stat_list(stat_paths) # see function below
        n_stat = len(stat_paths)

        # Ex
        if not n_stat == len(class_labels):
            raise ValueError('The number of trials do not match the number ' \
                             'of class labels')
        elif n_stat == 0:
            raise ValueError('There are no valid MNI COPES in ' + os.getcwd())
        
        # Pre-allocate
        mvpa_data = np.zeros([n_stat,n_features])

        # Load in data (COPEs)
        for i, path in enumerate(stat_paths):
            cope = nib.load(path).get_data()
            mvpa_data[i,:] = np.ravel(cope)[mask_index]

        # Normalization of voxel patterns by the variance corresponding to 
        # the raw beta weights        
        if norm_method == 'univariate':
            varcopes = glob.glob(sub_path + '/stats_new/varcope*mni.nii.gz')
            varcopes = sort_stat_list(varcopes)
            
            for i_trial, varcope in enumerate(varcopes):
                var = nib.load(varcope).get_data()
                var_sq = np.sqrt(var.ravel()[mask_index])
                mvpa_data[i_trial,] = mvpa_data[i_trial,] / var_sq
                
        if norm_method == 'multivariate':
            # This does not work yet. Should be implemented as multivariate
            # prewhitening as described in Walther et al. (unpublished draft),
            # http://www.icn.ucl.ac.uk/motorcontrol/pubs/representational_
            # analysis.pdf
            res4d = nib.load(sub_path + '/stats_new/res4d_mni.nii.gz').get_data()
            res4d.resize([np.prod(res4d.shape[0:3]), res4d.shape[3]])
            res4d = res4d[mask_index,]       
            #res_cov = np.cov(res4d)
        
        # Scale data (zero mean, unit variance) with sklearn scale function
        mvpa_data = preproc.scale(mvpa_data)
        
        # Initializing mvpa_mat object, which will be saved as a pickle file
        to_save = mvpa_mat(mvpa_data, sub_name, mask_name, mask_index, class_labels) 
        
        with open(mat_dir + '/' + sub_name + '.cPickle', 'wb') as handle:
            cPickle.dump(to_save, handle)
        
        if verbose:
            print "done."
        
def sort_stat_list(stat_list):
    """
    Sorts list with paths to statistic files (e.g. COPEs, VARCOPES),
    which are often sorted wrong (due to single and double digits).
    This function extracts the numbers from the stat files and sorts 
    the original list accordingly.
    """
    num_list = []    
    for path in stat_list:
        num = [str(s) for s in str(os.path.basename(path)) if s.isdigit()]
        num_list.append(int(''.join(num)))
    
    sorted_list = [x for y,x in sorted(zip(num_list, stat_list))]
    return(sorted_list)
    
def merge_runs(verbose):
    """ Merges single trial data from different runs. This function is 
    specifically written for the data from the Decoding Emotions subject
    (internship Research Master Psychology, Lukas Snoek), so it's not meant
    to be a generic merge function (only works in with this type of data). 
    Merges single trial data and initializes it again as mvap_mat object.
    """
    
    sub_paths = [os.path.abspath(path) for path in glob.glob(os.getcwd() + '/mvpa_mats/*WIPPM*cPickle*')]
    abbr = [os.path.basename(path)[6] for path in sub_paths]    
    sub_paths = [x for y,x in sorted(zip(abbr, sub_paths))]
    
    n_sub = len(sub_paths)
       
    i = 0
    # Loop over subjects 
    for dummy in xrange(n_sub/2):
        run1 = cPickle.load(open(sub_paths[i]))
        run2 = cPickle.load(open(sub_paths[i+1]))
        
        data = np.zeros((run1.n_trials*2, run1.n_features))
        class_labels = []
        
        if verbose:
            print "Merging subject %i ... " % (dummy+1),
        
        j = 0
        # Loop over trials, load data and class_labels
        for k in xrange(run1.n_trials-1):
            data[j,:] = run1.data[k,:]
            data[j+1,:] = run2.data[k,:]
                        
            class_labels.append(run1.class_labels[k])
            class_labels.append(run2.class_labels[k])
            
            j += 2
        
        class_labels.append(run1.class_labels[k+1])
        class_labels.append(run2.class_labels[k+1])
        
        # Create necessary class variables, initialize, and save
        name = os.path.basename(sub_paths[i])[0:7] + '_merged'
        mask_name = run1.mask_name
        mask_index = run1.mask_index
        
        to_save = mvpa_mat(data,name,mask_name, mask_index, class_labels)
        
        with open(os.getcwd() + '/mvpa_mats/' + name + '.cPickle', 'wb') as handle:
            cPickle.dump(to_save, handle)
            
        i += 2
        
        if verbose:
            print "done."
            
##############################################################################

#################### 4. ANALYSIS FUNCTIONS ###################################
""" These functions contain code for the multivariate encoding analysis and
assumes that pre-analysis processing has been performed. It needs an mvpa_mats 
directory with cPickle files containing mvpa_mat objects that can be processed. 
"""
   
class RDM():
    """ Object RDM, with averaged (over subjects) RDM as .data and other 
    attributes with information about the mask and classes. Idea for the
    future: inherit from mvpa_mat. Now, inherits from object.
    """
    
    def __init__(self, data, tval, mask_name, mask_index, class_labels):
        self.data = data
        self.tval = tval
        self.mask_name = mask_name
        self.mask_index = mask_index        
        self.class_labels = class_labels        
        self.n_trials = self.data.shape[0]
        class_names = []
        [class_names.append(i) for i in class_labels if not class_names.count(i)]
        self.class_names = class_names
        self.n_class = len(class_names)        
        self.n_inst = self.n_trials / self.n_class
        self.class_idx = [np.arange(self.n_inst*i,self.n_inst*i+self.n_inst) \
                          for i in range(self.n_class)]
    
def select_features(mvpa_data, prop_train, z_thres):
    n_train = np.round(mvpa_data.n_inst * prop_train)
    test_idx = np.zeros(mvpa_data.n_trials)
    
    for c in xrange(mvpa_data.n_class):
        ind = np.random.choice(mvpa_data.class_idx[c], n_train, replace=False)            
        test_idx[ind-1] = 1
        
    test_idx = test_idx.astype(bool)
    train_idx = np.invert(test_idx)   

    n_test = np.sum(train_idx) / mvpa_data.n_class
    trial_idx = [range(n_test*x,n_test*(x+1)) for x in range(mvpa_data.n_class)]
    data = mvpa_data.data[train_idx,] # should be moved to main_classify() later
    av_patterns = np.zeros((mvpa_data.n_class, mvpa_data.n_features))
    
    # Calculate mean patterns
    for c in xrange(mvpa_data.n_class):         
        av_patterns[c,] = np.mean(data[trial_idx[c],], axis = 0)
       
    # Create difference vectors, z-score standardization, absolute
    comb = list(itertools.combinations(range(1,mvpa_data.n_class+1),2))
    diff_patterns = np.zeros((len(comb), mvpa_data.n_features))
        
    for i, cb in enumerate(comb):        
        x = np.subtract(av_patterns[cb[0]-1,], av_patterns[cb[1]-1,])
        diff_patterns[i,] = np.abs((x - x.mean()) / x.std()) 
    
    diff_vec = np.mean(diff_patterns, axis = 0)
    feat_idx = diff_vec > z_thres

    return(feat_idx,test_idx)
    
def create_RDM(mvpa_data, prop_train, z_thres, method = 'euclidian'):
    """ Creates a symmetric distance matrix for an mvpa_mat object. Right now,
    uses the euclidian distance by default, but will (should?) be extended
    to incorporate different distance metrics. If no feature selection is
    performed (and thus no cross-validation), the RDM is created on the basis
    of all trials (i.e. when prop_train = 1). If feature-selection is desired,
    (when prop_train < 1), selection_features() is called on the train subset
    of the data; then, the RDM is created on the test subset.
    
    Args:
        mvpa_data:         instance of mvpa_mat class
        prop_train:        proportion of data to train on; if 1, then
                           no feature selection/cross-validation is performed
        z_thres:           Threshold for differentiation scores if 
                           prop_train < 1
        method:            distance metric; euclidian by defealt
    
    Returns:
        dist_mat:          Symmetrical distance matrix of prop_train * trials
                           (e.g. if prop_train = 0.5 and trials = 60, then
                           the dist_mat would be of size 30 x 30)
        test_idx:          Bool index of which trials belong to the test-set.
    """
    
    # If feature-selection is desired, call select_features()
    if prop_train < 1:
        feat_idx, test_idx = select_features(mvpa_data, prop_train, z_thres)    
    else:
        feat_idx = np.ones(mvpa_data.n_features).astype(bool)
        test_idx = np.ones(mvpa_data.n_trials).astype(bool)
    
    # Calculate pairwise euclidian distances
    if method == 'euclidian':
        dist_mat = euc_dist(mvpa_data.data[test_idx,:][:,feat_idx])
    elif method == 'mahalanobis':
        pass
        # Not yet incorporated, some trouble with NANs 
        
    return(dist_mat,test_idx)
        
def create_regressors(mvpa_data, test_idx, plot):
    """ 
    Creates RDM regressors/predictors given the factors in class_labels 
    attribute of mvpa_mat object. Is 'blind' to the amount of factors,
    viz. it works for both simple designs (1 factor) and designs with 
    more than one factor ('factorial' design). In the latter case, it 
    automatically creates an interaction predictor. Currently works with
    any amount of levels for each factor, but is limited by two factors +
    interaction term (in the future, it should be extended to accomodate
    more factors). Also plots the predictor RDMs using imshow.
    
    Args:
        mvpa_data:    An mvpa_mat object with, at least, .data and 
                      .class_labels attribute.
        test_idx:     Bool array with indices of test trials. If no feature
                      selection has been performed, it is an array of ones.
        plot:         if 1: plot regressor RDMs.        
    
    Returns:
        pred_RDM:     An ndarray of factors x RDM row x RDM col. For factorial
                      designs, the interaction term is included as the last
                      factor.
    """
    
    # Check whether it is a 'simple' (1 factor) or 'multiple' design (>1 factors)
    if any(isinstance(i, tuple) for i in mvpa_data.class_labels):
        n_fact = len(mvpa_data.class_labels[0])+1
        design = 'multiple'
    else:
        n_fact = 1
        design = 'simple'
    
    # Update parameters after feature selection
    n_trial = np.sum(test_idx)
    class_labels = list(np.array(mvpa_data.class_labels)[test_idx])
    
    # Pre-allocation of pred_RDM for speed
    pred_RDM = np.ones((n_fact,n_trial,n_trial))
    
    # Create main effect RDMs for factorial design
    if design == 'multiple':
        # Loop over factors
        for fact in xrange(n_fact-1):    
            entries = zip(*class_labels)[fact]
            
            # Loop over trials within factor
            for idx,trial in enumerate(entries):
                
                # This is quite nifty: compares trial class to the rest and 
                # returns inverted boolean vector (because if same, dist = 0).
                same = [trial == x for x in entries] 
                pred_RDM[fact,idx,:] = np.invert(same)
    
    # Create main effect for simple design; had to be in different block
    # because of indexing (could be more clean; future implementation!)
    else:
        entries = class_labels
        for idx,trial in enumerate(entries):
            same = [trial == x for x in entries]
            pred_RDM[0,idx,:] = np.invert(same)                  
            
    # Create interaction RDM if design is multiple
    if design == 'multiple':
        zipped = [str(x) + '_' + str(y) for x,y in class_labels]
        for idx,trial in enumerate(zipped):
                same = [trial == x for x in zipped]
                pred_RDM[-1,idx,:] = same
    
    # Plot predictor RDMs to check whether it makes sense
    if plot:    
        plot_RDM(design, n_fact, pred_RDM)
    
    return(pred_RDM)

def plot_RDM(design, n_fact, pred_RDM):
    """ 
    Plots predictor RDMs given pred_RDM. Is called within create_regressors(). 
    
    Args:
        design:    'simple' or 'multiple'
        n_fact:    number of factors (including interaction)
        pred_RDM:  predictor RDMs (ndarray)
        
    Returns: none.
    """

    if design == 'simple':
        n_plot = 1
    else:
        n_plot = n_fact
            
    for plot in xrange(n_plot):
        
        plt.subplot(1, n_plot, plot+1)
        plt.imshow(pred_RDM[plot,:,:], origin='lower')
        
        if plot != (n_plot-1) or plot == 0:
            plt.title('Factor ' + str(plot+1))
        else:
            plt.title('Interaction')
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'predictor_RDM.png'))
        
def test_RDM(observed, predictors):
    """
    test_RDM regresses the observed RDM onto the predictor RDMs as created 
    by create_regressors(). The lower triangular is first extracted and
    vectorized. The observed values are demeaned before regression. 
    
    Args:
        observed:    observed RDM, as returned by create_RDM()
        predictors:  predictor RDMs (ndarray, factors x RfDM row x RDM col)
    
    Returns:
        t_val:      (Array with) t_value(s) of length(factors)
        p_val:      P-value(s) associated with factor(s)
    """
        
    y = get_lower_half(observed)

    # Transform X from ndarray (fact x RDM row x RDM col) to 2D array
    # (features x factors), which can be regressed.    
    X = [predictors[i,:,:] for i in range(predictors.shape[0])]    
    
    if len(X) > 1:    
        X = [get_lower_half(item) for item in X] 
        X = np.vstack(tuple(X)).T
    else:
        X = get_lower_half(np.squeeze(predictors))
        X = X[:, np.newaxis]
    
    # Changed manual implementation to statsmodels implemenation due to
    # speed increase
    y = y - np.mean(y) # demeaning to avoid fitting intercept
    model = smf.OLS(y,X)
    results = model.fit() 
    t_val = float(results.tvalues)
    p_val = float(results.pvalues)
    
    return(t_val, p_val)

def get_lower_half(mat):
    """ Returns the lower triangular WITHOUT diagonal for a given matrix """
    idx = np.tril_indices(np.sqrt(mat.size), -1)
    return(mat[idx])
    
def permute_RDM(avRDM, predictors, iterations):
    
    t_vals = np.zeros(iterations)
    p_vals = np.zeros(iterations)
    y = get_lower_half(euc_dist(mvpa_data.data))
    y = y - np.mean(y) # demeaning to avoid fitting intercept
    X = [predictors[i,:,:] for i in range(predictors.shape[0])]        
    
    if len(X) > 1:    
        X = [get_lower_half(item) for item in X] 
        X = np.vstack(tuple(X)).T
    else:
        X = get_lower_half(np.squeeze(predictors))
        X = X[:, np.newaxis]
    
    for it in xrange(iterations):    
        np.random.shuffle(X)

        model = smf.OLS(y,X)
        results = model.fit() 
        t_vals[it] = float(results.tvalues)
        p_vals[it] = float(results.pvalues)
    
    return(t_vals, p_vals)
    
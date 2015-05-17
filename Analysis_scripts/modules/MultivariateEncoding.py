# -*- coding: utf-8 -*-
""" Multivariate Encoding module, created for the Research Master course
Programming: The Next Step, at the University of Amsterdam by Lukas Snoek.
This module contains the source code to run a multivariate encoding analysis
on single trial fMRI data. It is meant to work with first level FSL data.
It is by no means complete or perfectly 'generic', but works for simple 
1 factor data and provides initial functionality for factorial designs (but
is not thoroughly tested because there was no factorial data available). 

Dependencies (all open source):
- numpy
- scikit-learn 
- matplotlib
- nibabel
- nipy (for 2D plotting)



Lukas Snoek, spring 2015
"""

_author_ = "Lukas Snoek"

#################### Importing neccesary packages ############################

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

##############################################################################
'''
Create some random data set with two factors:
factor_let (A, B) and factor_num (1, 2) with each 15 instances.
Initilialize this as a mvpa_mat object (see glm2mvpa module).
'''

#data = np.random.normal(10, 5, (60, 100))
#subject_name = 'test'
#mask_name = 'random_gauss'
#mask_index = None
#mask_shape = None
#factor_let = ['a'] * 30 + ['b'] * 30
#factor_num = [1] * 15 + [2] * 15 + [1] * 15 + [2] * 15
#class_labels = zip(factor_let, factor_num)

#test_fac = mvpa_mat(data, subject_name, mask_name, mask_index, mask_shape, class_labels)
#obs_fac = create_RDM(test_fac)
#pred_fac = create_regressors(test_fac)

#os.chdir('/media/lukas/Data/Matlab2Python/FirstLevel/mvpa_mats/')
#test_simple = cPickle.load(open('HWW_001-20140205-0005-WIPPM_Zinnen1.cPickle'))
#obs_simple = create_RDM(test_simple)
#pred_simple = create_regressors(test_simple)
#tval = test_RDM(obs_simple, pred_simple)

#################### PREPROCESSING SCRIPTS ###################################
""" These functions (and class) take care of the 'preprocessing' of single
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
    
def create_subject_mats(mask,subject_stem,mask_threshold,norm_method = 'univariate'):
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
        
        print 'Processing ' + sub_name + ' ... ',
        
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
    
        print 'done.'
    print 'Created %i MVPA matrices' %  len(subject_dirs)

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
    
def merge_runs():
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
        print "Merged subject %i " % dummy
    
##############################################################################

#################### ANALYSIS SCRIPTS ########################################
""" These functions contain code for the multivariate encoding analysis and
assumes that preprocessing has been performed. It needs an mvpa_mats directory
with cPickle files containing mvpa_mat objects that can be processed. The main
analysis analysis function, main(), needs to be executed in the FirstLevel
directory. It loops over a specified amount of FSL (MNI) masks and subjects.
"""

def main(pickle_stem, method):
    
    # Initialize MNI ndarray to save t-values
    MNI_path = '/media/lukas/Data/Matlab2Python/FSL_masks/MNI152_T1_2mm_brain.nii.gz'    
    affine = MNI.get_affine()
    
    test = np.zeros((91,109,91))
    test[mask_index.reshape((91,109,91))] = 1
    thresholded_map = np.ma.masked_less(test, 0.5)
    
    plot_map(thresholded_map, MNI.get_affine())    
    
    sub_paths = [os.path.abspath(path) for path in glob.glob('*' + pickle_stem + '*.cPickle')]
    mean_RDM = np.zeros((len(sub_paths), 120, 120))    
    
    for i,sub in enumerate(sub_paths):
        mvpa_data = cPickle.load(open(sub))
        mean_RDM[i,:,:] = create_RDM(mvpa_data)
        #Reg = create_regressors(mvpa_data)
        #t_vals.append(test_RDM(RDM, Reg))
        print "Processed subject %i" % i + 1
    
    avRDM = np.mean(mean_RDM, axis = 0)
    Reg = create_regressors(mvpa_data)
    test_RDM(avRDM, Reg)
    
def create_RDM(mvpa_data, method = 'euclidian'):
    """ Creates a symmetric distance matrix for an mvpa_mat object. Right now,
    uses the euclidian distance by default, but will (should?) be extended
    to incorporate different distance metrics.
    """
    
    if method == 'euclidian':
        dist_mat = euc_dist(mvpa_data.data)
    elif method == 'mahalanobis':
        pass
        # Not yet incorporated, some trouble with NANs 
        
    return(dist_mat)
        
def create_regressors(mvpa_data):
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
    
    # Pre-allocation of pred_RDM for speed
    pred_RDM = np.ones((n_fact,mvpa_data.n_trials,mvpa_data.n_trials))
    
    # Create main effect RDMs for factorial design
    if design == 'multiple':
        # Loop over factors
        for fact in xrange(n_fact-1):    
            entries = zip(*mvpa_data.class_labels)[fact]
            
            # Loop over trials within factor
            for idx,trial in enumerate(entries):
                
                # This is quite nifty: compares trial class to the rest and 
                # returns inverted boolean vector (because if same, dist = 0).
                same = [trial == x for x in entries] 
                pred_RDM[fact,idx,:] = np.invert(same)
    
    # Create main effect for simple design; had to be in different block
    # because of indexing (could be more clean; future implementation!)
    else:
        entries = mvpa_data.class_labels
        for idx,trial in enumerate(entries):
            same = [trial == x for x in entries]
            pred_RDM[0,idx,:] = np.invert(same)                  
            
    # Create interaction RDM if design is multiple
    if design == 'multiple':
        zipped = [str(x) + '_' + str(y) for x,y in mvpa_data.class_labels]
        for idx,trial in enumerate(zipped):
                same = [trial == x for x in zipped]
                pred_RDM[-1,idx,:] = same
    
    # Plot predictor RDMs to check whether it makes sense
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
        
    fig = plt.figure()
    for plot in xrange(n_plot):
        plt.subplot(1, n_plot, plot+1)
        plt.imshow(pred_RDM[plot,:,:])
        if plot != (n_plot-1) or plot == 0:
            plt.title('Factor ' + str(plot+1))
        else:
            plt.title('Interaction')
    plt.colorbar()
        
def test_RDM(observed, predictors):
    """
    test_RDM regresses the observed RDM onto the predictor RDMs as created 
    by create_regressors(). The lower triangular is first extracted and
    vectorized. The observed values are demeaned before regression. 
    
    Args:
        observed:    observed RDM, as returned by create_RDM()
        predictors:  predictor RDMs (ndarray, factors x RfDM row x RDM col)
    
    Returns:
        t_vals:      Array with t_values of length(factors)
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
    
    # Manual implementation of multiple regression:
    # beta = (X'X)^-1 X'y 
    y = y - np.mean(y) # demeaning to avoid fitting intercept
    coeff = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)),X.T), y)
    y_hat = np.dot(coeff, X.T) # y-predicted
    
    # Calculating standard error of coefficient:
    # SE = sqrt(MSE * diag((X'X)^-1))
    MSE = np.mean((y - y_hat)**2)
    var_est = MSE * np.diag(np.linalg.pinv(np.dot(X.T,X)))
    SE_est = np.sqrt(var_est)    
    
    # t-values for coefficients (array)
    t_vals = coeff / SE_est
    
    return(t_vals)

def get_lower_half(mat):
    """ Returns the lower triangular WITHOUT diagonal for a given matrix """
    idx = np.tril_indices(np.sqrt(mat.size), -1)
    return(mat[idx])
    
   
    
    
    
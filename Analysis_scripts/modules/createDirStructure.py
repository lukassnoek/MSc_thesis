# -*- coding: utf-8 -*-
"""
This module contains functions to set-up the analysis directory structure
for preprocessing. It assumes that there is a project directory (project_dir)
in which all files (.PAR, .REC, .phy, .log) are dumped. This can be from
multiple subjects and (within-subject) runs at the same time. A ToProcess 
directory is created with subject-specific subdirecties, which in turn 
contain directories for separate "runs" (e.g. T1, func_A, func_B1, func_B2). 

Lukas Snoek, University of Amsterdam
- created for the Dynamic Affect project, supervised by Dr. H.S. Scholte
"""

import os
import glob
import shutil
import cPickle
import nibabel as nib

def convert_parrec2nifti(remove_nifti = 1, backup = 1):
    """ Loads in par/rec files using nibabel's parrec module and saves
    it as a nifti file. Temporary (?) fix because parrec2nifti commandline
    tool gives scaling error."""

    REC_files = glob.glob('*.REC')
    PAR_files = glob.glob('*.PAR')
    
    # Create scaninfo from PAR and convert .REC to nifti
    for REC,PAR in zip(REC_files, PAR_files):

        create_scaninfo(PAR)    
        REC_name = REC[:-4]
        
        if not os.path.exists(REC_name + '.nii'):
            print "Processing file %s" % (REC_name + ' ...'),              
            PR_obj = nib.parrec.load(REC)        
            nib.nifti1.save(PR_obj,REC_name)
            print " done."
            
        else:
            print "File %s was already converted." % (REC_name)
     
    # Zipping to .gz.nii format
    print "Zipping niftis ... ",     
    os.system("gzip *.nii")
    print "done."
    
    if remove_nifti:
        os.system('rm *.nii')
    
    if backup:
        backup_parrec()
        
#def convert_parrec2nii(directory, remove_nifti = 1):
#    """Converts Philips PAR/REC files to nifti files"""
    
#    os.chdir(directory)
#    os.system("parrec2nii *.PAR") # deprecated
#    os.system("gzip *.nii")
    
#    PAR_files = glob.glob('*.PAR')
#    n_PAR = len(PAR_files)
#    print "Converted {0} files".format(n_PAR)
    
    # Extract useful info from PAR-header
    
#    for PAR in PAR_files:
#        create_scaninfo(PAR)

#    if remove_nifti:
#        for file in glob.glob('*.nii'):
#            os.remove(file)

def create_scaninfo(par_filepath):
    """ Docstring """
    
    # Open .par file
    #txt = [line.rstrip('\n').rstrip('\r') for line in open(par_filepath)]
    
    # Empty dic
    #scaninfo = {}
    
    # Loop over lines in par-file
    #for i,line in enumerate(txt):
    #    if len(line) > 0:
            
            # Only extract info from lines starting with "."
   #         if line[0] == '.':
   #             key = line.split(":")[0].rstrip()[5:]            
   #             values = line.split(":")[1]            
   #             values = [x for x in values.split(" ") if len(x) > 0]
            
                # Loop over values to check if float or otherwise
   #             for i,val in enumerate(values):
   #                 try:
   #                     val = float(val)
   #                 except (ValueError, TypeError):
   #                     pass
                
   #                values[i] = val
                
   #             n_str = [type(x) == str for x in values]                
   #             if sum(n_str) == len(values):
   #                 values = " ".join(values)
                        
                # Save key-value pair in dictionary
   #             scaninfo[key] = values
    
    fID = open(par_filepath)
    scaninfo = nib.parrec.parse_PAR_header(fID)[0]
    fID.close()
    
    to_save = os.path.join(os.getcwd(), + par_filepath[:-4] + '_scaninfo' + '.cPickle')
    
    with open(to_save,'wb') as handle:
            cPickle.dump(scaninfo, handle)
  
def backup_parrec():
    """Create back-up dir with all original par/recs"""
    
    backup_dir = os.getcwd() + '/rawdata_backup'

    if not os.path.exists(backup_dir):
        os.mkdir(backup_dir)

    PAR_files = glob.glob('*.PAR')
    REC_files = glob.glob('*.REC')

    to_move = zip(PAR_files, REC_files)
    
    for PAR,REC in to_move:
        shutil.move(PAR, backup_dir)
        shutil.move(REC, backup_dir)
        
    print "Back-up completed for %i files" % (len(to_move))

def move_files(from_dir, to_dir, keyword):
    """
    Moves files, containing the given keyword, from a specified directory
    (from_dir) to another (to_dir). If keyword is None, from_dir is 
    treated as a list of files (absolute paths), which can be from different
    directories, which are subsequently moved to to_dir.
    """
    
    if not os.path.exists(to_dir):
        os.mkdir(to_dir)
    
    if keyword == None:
        # If keyword is left empty, from_dir is considered a list of files.
        to_move = from_dir
    else:
        to_move = glob.glob(os.path.join(from_dir, '*' + keyword + '*'))
    
    n_moved = 0    
    for f in to_move:
        if os.path.isfile(f):
            shutil.move(f, to_dir)
            n_moved += 1
    
    print "Moved %i files to %s." % (n_moved, to_dir)

def setup_analysis_skeleton(project_dir,sub_stem):
    """
    Moves files from ToProcess to subject-specific directories
    """
    
    # Move from project_dir to ToProcess
    TP_dir = os.path.join(project_dir,'ToProcess')
    move_files(project_dir, TP_dir, '')
    
    all_files = glob.glob(TP_dir + '/*' + sub_stem + '*')
    all_files = [os.path.basename(f) for f in all_files]
    
    prefixes = [pref[:len(sub_stem)+4] for pref in all_files]
    subjects = list(set(prefixes))
    subjectDir = []

    for idx, subs in enumerate(subjects):
    
        subjectDir.append(subs[0:len(subject_stem) + 3])
    
        if os.path.exists(subjectDir[idx]) == 0:
            os.mkdir(subjectDir[idx])
    
        toMove = glob.glob(subs[0:len(subject_stem) + 3] + '*')
    
        # Move all files to subject-specific dirs
        for mFile in toMove:
            if os.path.isfile(mFile):
                shutil.move(mFile, subjectDir[idx])
            
            for entry in glob.glob('*'):
                if os.path.isfile(entry):
                    print "Not allocated: " + entry
                    
    subject_dir_list = [os.getcwd() + '/' + sDir for sDir in subjectDir]
    
def reset_pipeline(project_dir, except_nifti = 1):
    """
    Resets to analysis set-up to raw files aggregated in the project_dir. 
    Retrieves log/phy files from ToProcess and PAR/REC from backup_dir. 
    Subsequently removes all subdirectories of the project_dir.
    """
    
    answer = raw_input("Are you sure you want to reset? (Y/N): ")
    
    if answer.upper() == 'Y':
        
        # Find and move log and phy files
        log_files, dummy = get_filepaths('.log',project_dir)
        phy_files, dummy = get_filepaths('.phy', project_dir)    
        
        # Pop files which are already in project_dir
        log_tomove = []
        phy_tomove = []
        
        for log in log_files:
            if not os.path.dirname(log) == project_dir:
                log_tomove.append(log)
        
        for phy in phy_files:
            if not os.path.dirname(phy) == project_dir:
                phy_tomove.append(phy)            
        
        move_files(log_tomove, project_dir, None)
        move_files(phy_tomove, project_dir, None)
    
        # Move par/rec files from backup to project_dir
        backup_dir = os.path.join(project_dir,'rawdata_backup')
        move_files(backup_dir,project_dir, '*')
    
        # Remove directories
        [os.system('rm -R ' + dirx) for dirx in glob.glob(project_dir + '/*') \
            if os.path.isdir(dirx)]        
        
        # Remove cPickle
        [os.system('rm ' + cPick) for cPick in glob.glob(project_dir + '/*.cPickle')]

        if not except_nifti:
            [os.system('rm ' + nii) for nii in glob.glob(project_dir + '/*.nii*')]

        print "Pipeline has been reset!"
        
def movefiles_subjectdirs(subject_stem, ToProcess):
    """
    Moves files to subject specific directories and creates run-specific
    subdirectories
    """
    
    os.chdir(ToProcess)
    allFiles = glob.glob(subject_stem + '*')
    prefixes = [pref[0:7] for pref in allFiles]
    subjects = list(set(prefixes))
    subjectDir = []

    for idx, subs in enumerate(subjects):
    
        subjectDir.append(subs[0:len(subject_stem) + 3])
    
        if os.path.exists(subjectDir[idx]) == 0:
            os.mkdir(subjectDir[idx])
    
        toMove = glob.glob(subs[0:len(subject_stem) + 3] + '*')
    
        # Move all files to subject-specific dirs
        for mFile in toMove:
            if os.path.isfile(mFile):
                shutil.move(mFile, subjectDir[idx])
            
            for entry in glob.glob('*'):
                if os.path.isfile(entry):
                    print "Not allocated: " + entry
                    
    subject_dir_list = [os.getcwd() + '/' + sDir for sDir in subjectDir]
    
    # Create subdirectories
    for subjectDir in subject_dir_list:
        os.chdir(subjectDir)
    
        mri_files = glob.glob('*.nii.gz')
        subdir_names = []
        
        for mriFile in mri_files:
            split_file = mriFile.split('_')
            from_idx = split_file.index('WIP')
            to_idx = split_file.index('SENSE')
            toAppend = "_".join(split_file[from_idx+1:to_idx])            
            subdir_names.append(toAppend)
            
            os.mkdir(toAppend)
            shutil.move(mriFile, toAppend)
        
        print "Created the following subdirs for {0}: ".format(os.path.basename(subjectDir))
        for subdir in subdir_names:
            print subdir
        print "\n"
        
def get_filepaths(keyword, directory):
    """ 
    Given a certain keyword (including wildcards), this function walks
    through subdirectories relative to arg directory (i.e. root) and returns 
    matches (absolute paths) and filenames_total (filenames).
    """
     
    matches = []
    filenames_total = []
    
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if keyword in filename:
                matches.append(root + '/' + filename)
                filenames_total.append(filename)
    return matches, filenames_total
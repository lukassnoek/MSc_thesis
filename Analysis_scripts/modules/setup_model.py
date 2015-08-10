# -*- coding: utf-8 -*-
"""
Module with functions to create .bfsl files from Presentation-logfiles
and consequently set up a FEAT first level model
"""

# TO DO: 
# UNIVARIATE OPTION (for generate_FEAT_info)

# --------------------------- IMPORTS --------------------------- #
import numpy as np
import os
import pandas as pd
import glob
import nibabel as nib

# --------------------------- highres2func --------------------- #
def transform_highres2func(T1_path,reg_path):
    
    transform_mat = os.path.join(reg_path,'reg','highres2example_func.mat')
    ref_func = glob.glob(os.path.join(reg_path,'func_data*.nii.gz'))
        
    csf_file = glob.glob(os.path.join(T1_path,'pve_0.nii.gz'))
    graymatter_file = glob.glob(os.path.join(T1_path,'pve_1.nii.gz'))
    whitematter_file = glob.glob(os.path.join(T1_path,'pve_2.nii.gz'))    
    
    flirt_csf = 'flirt -in %s -ref %s, -applyxfm -init %s -out %s' % \
    (csf_file,ref_func,transform_mat, 'csf_mask')
    
    flirt_graymatter = 'flirt -in %s -ref %s, -applyxfm -init %s -out %s' % \
    (graymatter_file,ref_func,transform_mat, 'graymatter_mask')
    
    flirt_whitematter = 'flirt -in %s -ref %s, -applyxfm -init %s -out %s' % \
    (whitematter_file,ref_func,transform_mat, 'whitematter_mask')    

    flirt_list = [flirt_csf, flirt_graymatter, flirt_whitematter]
    
    for flirt in flirt_list:
        print "Tranforming %s"
        os.system(flirt)

# --------------------------- COVARS --------------------------- #
T1_file = os.path.join(data_dir,'FAST','da01-20150721-0007-T1_brain.nii.gz')
T1_path = os.path.dirname(T1_file)
func_file = os.path.join(data_dir,'FAST','func_data_mcst_sg_ss5.nii.gz')

# Segment!
fast_command = 'fast %s' (T1_file)
os.system(fast_command)

# Transform
transform_highres2func(os.path.dirname(T1_file),reg_dir)

# Mask
csf_mask = nib.load(os.path.join(T1_path,'csf_mask.nii.gz')).get_data() > .5
graymatter_mask = nib.load(os.path.join(T1_path,'graymatter_mask.nii.gz')).get_data() > .5
whitematter_mask = nib.load(os.path.join(T1_path,'whitematter_mask.nii.gz')).get_data() > .5

func_brain = nib.load(func_file).get_data()

csf_ts = np.mean(func_brain[csf_mask], axis = 0)
gm_ts = np.mean(func_brain[graymatter_mask], axis = 0)
wm_ts = np.mean(func_brain[whitematter_mask], axis = 0)

np.savetxt('csf_regressor.bfsl', csf_ts, newline=" ", fmt = '%1.3f')
np.savetxt('gm_regressor.bfsl', gm_ts, newline=" ", fmt = '%1.3f')
np.savetxt('wm_regressor.bfsl', wm_ts, newline=" ", fmt = '%1.3f')

plt.plot(csf_ts)
plt.plot(gm_ts)
plt.plot(wm_ts)
plt.show()

# --------------------------- Reverse regressor ---------------- #
def create_baseline_regressor(log_path, pulsecode):
    '''
    Creates a "baseline" regressor (i.e. a three-column bfsl-file
    for a first-level GLM of (slow event-related) fMRI data. Could
    be useful as a sanity check and perhaps as to decrease the 
    error term (useful in particular for single-trial analyses).
    
    Args:
        log_path (str): absolute path to Presentation (Neurobs) logfile
        pulsecode (int): logging-code for (first) fMRI pulse
        
    NOT YET FINISHED. ONLY WORKS FOR NARRATIVE RUN AND HAS ISSUES
    WITH TIMING RELATIVE TO FIRST PULSE.
    '''
    
    base_dir = os.path.dirname(log_path)    
    df = pd.read_table(log_path, sep = '\t',skiprows=3,header = 0, skip_blank_lines=True)

    # Convert to numeric and drop all rows until first pulse
    df = df.convert_objects(convert_numeric = True)
    pulse_idx = int((np.where(df['Code'] == pulsecode)[0]))
    df = df.drop(range(pulse_idx))

    # pulse_t = absolute time of first pulse
    pulse_t = df['Time'][df['Code'] == pulsecode]

    # Trials which should NOT be part of the baseline
    iso_stimtimes = np.array(df['Time'][df['Code'] < 101])

    # This is specific for the narrative run
    audio_onset = iso_stimtimes[1:] + 10
    audio_onset = np.append(0.0, audio_onset)
    audio_onset = np.delete(audio_onset,-1)

    audio_dur = np.append(iso_stimtimes[1],np.diff(iso_stimtimes[1:]) - 10)

    # Write to file
    to_write = pd.DataFrame()
    to_write['Time'] = audio_onset
    to_write['Duration'] = audio_dur
    to_write['Weight'] = np.ones(len(audio_onset))

    path_bfsl = '%s/%s.bfsl' % (base_dir,'narrative_audio')
    to_write.to_csv(path_bfsl, sep = '\t', header = False, index = False)

# --------------------------- SET UP --------------------------- #
# Idea: built in a way to generate a 'univariate' regressor (ignore)
con_names = ['Poschar','Negchar','Neuchar',
             'Posloc','Negloc','Neuloc',
             'Eval']
'''
con_codes = [[10],[11],[12],
             [20],[21],[22],
            [101,200]]
'''            

con_codes = [[10],[11],[12],
             [20],[21],[22],
            [101,200]]
'''
con_names = ['sit_pos','sit_neg','sit_neu',
             'fac_pos','fac_neg','fac_neu']

con_codes = [[11000],[21000],[31000],
             [12100,12200],[22100,22200],[32100,32200]]
'''
con_info = zip(con_names, con_codes)

#log_path = '/media/lukas/Data/DynamicAffect/da01_preproc/20150721-0003-pretest/da01-DynAff_pretest_vs123123.log'
log_path = os.path.join(home_dir, 'da01-DynAff_posttest_vs123123.log')
design = 'singletrial'
pulsecode = 100

# --------------------------- Pres2bfsl --------------------------- #
def generate_bfsl(log_path, con_info, design, pulsecode = 100):
    """ Generates bfsl-textfiles based on Presentation-logfile.

    Args:
        log_path: absolute path to logfile
        con_info: list of tuples with condition-info as (name, code)
        design: 'univar' (regressor per cond) or 'singletrial'
        pulsecode: code for (first) pulse
    """

    con_names, con_codes = zip(*con_info)
    base_dir = os.path.dirname(log_path)

    # Read in logfile as pandas df
    df = pd.read_table(log_path, sep = '\t',skiprows=3,
                       header = 0,
                       skip_blank_lines=True)

    # Convert to numeric and drop all rows until first pulse
    df = df.convert_objects(convert_numeric = True)
    pulse_idx = int(np.where(df['Code'] == pulsecode)[0])
    df = df.drop(range(pulse_idx))

    # pulse_t = absolute time of first pulse
    pulse_t = df['Time'][df['Code'] == pulsecode]
    df['Time'] = (df['Time']-float(pulse_t)) / 10000.0
    df['Duration'] = df['Duration'] / 10000.0

    n_con = []
    bfsl_paths = []

    # Loop over conditions and write .bfsl files with timings
    for i,code in enumerate(con_codes):
        to_write = pd.DataFrame()

        # Get index for trials with specific code
        if len(code) > 1:
            idx = df['Code'].isin(xrange(code[0],code[1]+1))
        else:
            idx = df['Code'] == code

        # Generate dataframe with time, duration, and weight given idx
        to_write['Time'] = df['Time'][idx]
        to_write['Duration'] = df['Duration'][idx]
        to_write['Weight'] = np.ones((sum(idx),1))

        # Write to txt-file
        if design == 'univar':
            path_bfsl = '%s/%s.bfsl' % (base_dir,con_names[i])
            to_write.to_csv(path_bfsl, sep = '\t',
                            header = False, index = False)
            bfsl_paths.append(path_bfsl)

        # Write to single txt-files
        elif design == 'singletrial':

            for j in xrange(to_write.shape[0]):
                to_write2 = to_write.iloc[[j]]
                path_bfsl = '%s/%s_00%i.bfsl' % (base_dir,con_names[i],(j+1))

                to_write2.to_csv(path_bfsl, sep = '\t',
                                 header = False, index = False)

                bfsl_paths.append(path_bfsl)
        n_con.append(sum(idx))
        # End for-loop over conditions

    # n_ev = number of
    if design == 'univar':
        n_ev = len(n_con)
    elif design == 'singletrial':
        n_ev = sum(n_con)

    generate_FEAT_info(base_dir,n_ev,con_names,n_con, bfsl_paths)

# --------------------------- bfsl2fsf --------------------------- #
def generate_FEAT_info(directory, n_ev, con_names, n_con, bfsl_paths):
    """
    Generates information about model setup, to-be read into
    FEAT.
    """

    infofile = os.path.join(directory, 'FEAT_EV_info.txt')

    if os.path.isfile(infofile):
        os.remove(infofile)

    out = open(os.path.join(directory, 'FEAT_EV_info.txt'), 'w')

    out.write('set fmri(evs_orig): %i' % (np.sum(n_con)))
    out.write('\n')
    out.write('fmri(evs_real): %i' % (np.sum(n_con)))
    out.write('\n')    
    out.write('fmri(ncon_orig): %i' % (np.sum(n_con)))
    out.write('\n')      
    out.write('fmri(ncon_real): %i' % (np.sum(n_con)))
    out.write('\n')

    c_ev = 1
    for c in xrange(len(n_con)):
        c_name = con_names[c]

        for ev in xrange(n_con[c]):

            out.write('set fmri(evtitle%i) "%s%i"' % (c_ev, c_name, ev+1))
            out.write('\n')
            out.write('set fmri(shape%i) 3' % (c_ev))
            out.write('\n')
            out.write('set fmri(convolve%i) 3' % (c_ev))
            out.write('\n')
            out.write('set fmri(convolve_phase%i) 0' % (c_ev))
            out.write('\n')
            out.write('set fmri(tempfilt_yn%i) 1' % (c_ev))
            out.write('\n')
            out.write('set fmri(deriv_yn%i) 0' % (c_ev))
            out.write('\n')
            out.write('set fmri(custom%i) "%s"' % (c_ev, bfsl_paths[c_ev-1]))
            out.write('\n')
            out.write('set fmri(conpic_orig.%i) 1' % (c_ev))
            out.write('\n')
            out.write('set fmri(conpic_real.%i) 1' % (c_ev))
            out.write('\n')
            out.write('set fmri(conname_orig.%i) "%s%i"' % (c_ev, c_name, ev+1))
            out.write('\n')
            out.write('set fmri(conname_real.%i) "%s%i"' % (c_ev, c_name, ev+1))
            out.write('\n')

            for n in xrange(1,n_ev+1):
                if n == (ev+1):
                    out.write('set fmri(con_orig%i.%i) 1' % (c_ev, n))
                    out.write('\n')
                    out.write('set fmri(con_real%i.%i) 1' % (c_ev, n))
                    out.write('\n')
                else:
                    out.write('set fmri(con_orig%i.%i) 0' % (c_ev, n))
                    out.write('\n')
                    out.write('set fmri(con_real%i.%i) 0' % (c_ev, n))
                    out.write('\n')
                    out.write('set fmri(conmask%i_%i) 0' % (c_ev, n))
                    out.write('\n')

                out.write('set fmri(ortho%i.%i) 0' % (c_ev, n))
                out.write('\n')
                out.write('set fmri(conmask%i_%i) 0' % (c_ev, n))
                out.write('\n')

            for n in xrange(n_ev+1):
                out.write('set fmri(ortho%i.%i) 0)' % (c_ev, n))
                out.write('\n')

            c_ev += 1
    out.close()

#--------------------------- set-up FEAT --------------------------- #
import numpy as np
import cPickle as cp

feat_params = {'fmri(outputdir)': '"%s"' % ('/home/c6386806/Desktop/DynamicAffect/da01_FL/stimdriv'),
'fmri(TR)': 2,
'fmri(npts)': 218,
'fmri(mc)': 1,
'fmri(temphp_yn)': 1,
'fmri(templp_yn)': 0,
'fmri(smooth)': 5,
'fmri(prewhiten_yn)': 1,
'fmri(thres)': 1,
'fmri(regstandard)': '"/usr/share/fsl/data/standard/MNI152_T1_2mm_brain"',
'feat_files(1)': '"/media/lukas/Data/DynamicAffect/da01_preproc/20150721-0006-posttest/da01-20150721-0006-posttest"',
'highres_files(1)': '"/media/lukas/Data/DynamicAffect/da01_preproc/20150721-0007-T1/da01-20150721-0007-T1_brain"'}

fsf_template = '/media/lukas/Data/DynamicAffect/generic_template.fsf'
fsf_info = '/media/lukas/Data/DynamicAffect/da01_preproc/20150721-0006-posttest/FEAT_EV_info.txt'

def create_fsf(fsf_template, feat_params, fsf_info):

    fsf = np.genfromtxt(open(fsf_template, 'r'), dtype='str')

    for key,value in feat_params.iteritems():

        idx = np.where(fsf == key)
        fsf[idx[0],2] = value

    base_name = os.path.dirname(fsf_template)

    filename = base_name + '/test.fsf'
    out = open(filename, 'w')

    for n in xrange(fsf.shape[0]):
        out.write(fsf[n,0])
        out.write(' ')
        out.write(fsf[n,1])
        out.write(' ')
        out.write(fsf[n,2])
        out.write('\n')

    out.close()
    os.system('cat %s >> %s' % (fsf_info, filename))
    
__author__ = 'lukas'

import os
from os.path import join as opj
import nibabel as nib
import glob
import numpy as np
import shutil

def main(zipped_stats, mapping, zval, pval):
    """ Runs conjunction analysis and extracts ROI info from conjunctions

    Args:
        zipped_stats (list): list of tuples of to-be-analyzed conjunctions;
            should be thresholded nifti files
        mapping (dict): mapping between cope-names and original contrast-names
        zval (float): z-value cutoff for cluster-correction
        pval (flaot): p-value minimum for cluster-correction

    """
    stat1, stat2 = zip(*zipped_stats)
    base_dir = opj(os.getcwd(), 'Conjunction_analysis')
    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)

    name1 = [os.path.basename(os.path.dirname(x)) for x in stat1]
    name2 = [os.path.basename(os.path.dirname(x)) for x in stat2]

    if len(np.unique(name1)) > 1:
        names = name1
    else:
        names = name2

    for i, path in enumerate(zipped_stats):
        stat1, stat2 = path

        tmp_name = names[i][:-5] + '_' + os.path.basename(stat1)[:-7]
        new_name = mapping[tmp_name]
        out_dir = opj(base_dir, new_name)

        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)

        os.mkdir(out_dir)
        os.chdir(out_dir)
        print "Processing %s" % out_dir

        os.system('cp %s %s' % (opj(os.path.dirname(base_dir), 'easythresh_conj.sh'), out_dir))
        mask = opj(os.path.dirname(stat1), 'example_func.nii.gz')
        background = mask

        cmd = 'bash easythresh_conj.sh %s %s %s %f %f %s conjunction' % \
              (stat1, stat2, mask, zval, pval, background)

        _ = os.system(cmd)

        mask_list = glob.glob(opj('/home/c6386806/ROIs/Harvard_Oxford_atlas/unilateral', '*nii.gz*'))

        extract_roi_info(opj(out_dir, 'thresh_conjunction.nii.gz'),
                         opj(out_dir, 'cluster_mask_conjunction.nii.gz'),
                         mask_list, 30, per_cluster=True)


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

    sorted_list = [x for y, x in sorted(zip(num_list, stat_list))]
    return(sorted_list)

if __name__ == '__main__':
    import sys
    sys.path.append('/home/c6386806/LOCAL/Analysis_scripts')
    from modules.extract_roi_info import main as extract_roi_info

    home = os.path.expanduser('~')
    univar_dir = opj(home, 'Desktop', 'DecEmo_uni')
    os.chdir(univar_dir)

    stats1 = glob.glob(opj(univar_dir, 'FSL_ThirdLevel_Zinnen_p005', '*', 'cope1.feat', 'thresh_zstat*.nii.gz'))
    stats2 = glob.glob(opj(univar_dir, 'FSL_ThirdLevel_HWW_p005.gfeat', '*', 'thresh_zstat*.nii.gz'))

    stats1 = sort_stat_list(stats1)
    stats2 = sort_stat_list(stats2)

    zipped_stats = zip(sort_stat_list(stats1), sort_stat_list(stats2))

    # Mapping copes >> contrasts
    mapping = {'cope1_thresh_zstat1': 'Action_min_baseline',
               'cope2_thresh_zstat1': 'Interoception_min_baseline',
               'cope3_thresh_zstat1': 'Situation_min_baseline',
               'cope4_thresh_zstat1': 'Action_min_Interoception',
               'cope5_thresh_zstat1': 'Action_min_Situation',
               'cope6_thresh_zstat1': 'Interoception_min_Situation',
               'cope7_thresh_zstat1': 'Action_min_other',
               'cope8_thresh_zstat1': 'Interoception_min_other',
               'cope9_thresh_zstat1': 'Situation_min_other',
               'cope10_thresh_zstat1': 'Stimulation_min_baseline',
               'cope1_thresh_zstat2': 'baseline_min_Action',
               'cope2_thresh_zstat2': 'baseline_min_Interoception',
               'cope3_thresh_zstat2': 'baseline_min_Situation',
               'cope4_thresh_zstat2': 'Interoception_Action',
               'cope5_thresh_zstat2': 'Situation_min_Action',
               'cope6_thresh_zstat2': 'Situation_Interoception',
               'cope7_thresh_zstat2': 'Other_min_Action',
               'cope8_thresh_zstat2': 'Other_min_Interoception',
               'cope9_thresh_zstat2': 'Other_min_Situation',
               'cope10_thresh_zstat2': 'baseline_min_Stimulation'}

    main(zipped_stats, mapping, 2.6, 0.05)

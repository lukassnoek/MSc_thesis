__author__ = 'lukas'


def optimize_clustering(sub_dir, inputs):
    iterations = inputs['iterations']
    n_test = inputs['n_test']
    fs_method = inputs['fs_method']
    fs_arg = inputs['fs_arg']
    cluster_min = inputs['cluster_min']
    score_unit = 0 if inputs['score_unit'] == 'TPR' else 1

    print "performing analysis with zval = %f and cluster_min = %i" % (fs_arg, cluster_min)
    # Definition of classifier
    clf = svm.LinearSVC()

    # Unpacking subject-data into header and actual data
    header_path, data_path = sub_dir
    mvp = cPickle.load(open(header_path))
    mvp.data = h5py.File(data_path, 'r')['data'][:]

    # Containers for classification data, tracked over iterations
    conf_mat = np.zeros((mvp.n_class, mvp.n_class))
    clusters = np.zeros(iterations)
    n_features = np.sum(mvp.mask_index)
    fs_data = {'count': np.zeros(n_features),
               'score': np.zeros(n_features)}
    vox_score = np.zeros(n_features)
    vox_idx = np.zeros(vox_score.shape)
    cluster_cleanup = False

    folds = sss(mvp.num_labels, iterations, n_test * mvp.n_class,
                      random_state=0)

    skip = 0
    for i, (train_idx, test_idx) in enumerate(folds):

        # Index data (X) and labels (y)
        train_data = mvp.data[train_idx, :]
        test_data = mvp.data[test_idx, :]
        train_labels = np.asarray(mvp.num_labels)[train_idx]
        test_labels = np.asarray(mvp.num_labels)[test_idx]

        selector = fs_method(fs_arg)
        selector.fit(train_data, train_labels)

        # Cluster feature selection, if specified
        if np.sum(selector.idx) == 0:
            final_score = 0
            clusters = 0
            fs_success = False
            cluster_success = False
            skip = 1
            break

        inpt = {'mvp': mvp, 'train_data': train_data,
                'test_data': test_data, 'fs_arg': fs_arg,
                'cluster_min': cluster_min, 'selector': selector,
                'fs_data': fs_data, 'vox_idx': vox_idx,
                'cluster_cleanup': cluster_cleanup}

        # Cluster data & return averaged (if not cluster_cleanup) ftrs
        output = clustercorrect_feature_selection(**inpt)
        train_data, test_data, cl_idx = output

        if train_data.shape[1] == 0:
            final_score = 0
            clusters = 0
            fs_success = True
            cluster_success = False
            skip = 1
            break

        clusters[i] = train_data.shape[1]

        clf.fit(train_data, train_labels)
        test_pred = clf.predict(test_data)
        conf_mat += confusion_matrix(test_labels, test_pred)

    if skip == 0:
        conf_mat = np.true_divide(conf_mat, np.sum(conf_mat, score_unit))
        final_score = np.mean(np.diag(conf_mat))
        fs_success = True
        cluster_success = True

    # Write out classification results as pandas dataframe
    df = {'sub_name': mvp.subject_name,
          'zval': fs_arg,
          'cluster_min': cluster_min,
          'n_clust': np.round(np.mean(clusters), 3),
          'cluster_success': cluster_success,
          'fs_success': fs_success,
          'score': np.round(final_score, 3)}
    df = pd.DataFrame(df, index=[0])

    fn = opj(os.getcwd(),'opt_clust', 'results_z%f_cmin%i_%s.csv' % (fs_arg, cluster_min, mvp.subject_name))
    with open(fn, 'w') as f:
        df.to_csv(f, header=True, sep='\t', index=False)

if __name__ == '__main__':
    import sys
    import time
    import pandas as pd
    from os.path import join as opj
    from sklearn.cross_validation import StratifiedShuffleSplit as sss
    import psutil
    sys.path.append('/home/c6386806/LOCAL/Analysis_scripts')

    from joblib import Parallel, delayed
    from modules.glm2mvpa import MVPHeader
    from modules.main_classify import *

    # Information about which data to use
    home = os.path.expanduser('~')
    feat_dir = opj(home, 'DecodingEmotions')
    ROI_dir = opj(home, 'ROIs')
    os.chdir(feat_dir)
    identifier = ''

    mvp_dir = opj(os.getcwd(), 'mvp_mats')
    header_dirs = sorted(glob.glob(opj(mvp_dir, '*%s*cPickle' % identifier)))
    data_dirs = sorted(glob.glob(opj(mvp_dir, '*%s*hdf5' % identifier)))
    subject_dirs = zip(header_dirs, data_dirs)

    # Parameters for classification
    inputs = {}
    inputs['iterations'] = 100
    inputs['n_test'] = 4
    inputs['fs_method'] = SelectAboveZvalue
    inputs['fs_arg'] = 2.3
    inputs['cluster_min'] = 10
    inputs['score_unit'] = 'PPV'
    inputs['do_pca'] = False

    debug = False
    n_proc = 1 if debug else len(subject_dirs)

    while True:
        cpu_use = psutil.cpu_percent()

        if cpu_use > 50:
            print "CPU utilization at %f percent." % cpu_use
            time.sleep(10)
        else:
            "Starting analysis because CPU is at %f." % cpu_use
            break

    for fs_arg in np.arange(1, 2.75, 0.25):
        inputs['fs_arg'] = fs_arg
        for cluster_min in np.arange(10, 311, 25):
            inputs['cluster_min'] = cluster_min
            Parallel(n_jobs=n_proc) \
                (delayed(optimize_clustering)(sub_dir, inputs) for sub_dir in subject_dirs)


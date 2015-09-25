"""
Script to load in accuracy-scores per subject and to calculate effect sizes
for different sample sizes using a bootstrap procedure.
"""
__author__ = 'lukas'

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from os.path import join as opj
from scipy.stats import t

def main(filepath, iterations):
    data = pd.read_csv(filepath, sep=',', skiprows=22)
    data = data.dropna(how='any')
    scores = np.asarray(data['accuracy'])

    sample_size = scores.shape[0] + 1

    df = {'mean': np.zeros(sample_size - 2),
          'npval': np.zeros(sample_size - 2),
          'std': np.zeros(sample_size - 2),
          'pval': np.zeros(sample_size - 2),
          'nsub': np.zeros(sample_size - 2),
          'cohen': np.zeros(sample_size - 2)}

    for j, nsub in enumerate(xrange(2, scores.shape[0] + 1)):

        mean_class = np.zeros(iterations)
        std_class = np.zeros(iterations)

        for i in xrange(iterations):
            tmp = scores[np.random.randint(scores.shape[0], size=nsub)]
            mean_class[i] = np.mean(tmp)
            std_class[i] = np.std(tmp)

        tvals = (mean_class - 0.3333) / (std_class / np.sqrt(nsub-1))
        pvals = [t.sf(np.abs(tt), nsub-1) * 2 for tt in tvals]

        df['npval'][j] = np.sum(np.asarray(pvals) > 0.05) / float(iterations)
        df['mean'][j] = np.mean(mean_class)
        df['std'][j] = np.std(np.asarray(pvals))
        df['pval'][j] = np.mean(np.asarray(pvals))
        df['nsub'][j] = nsub
        df['cohen'][j] = (np.mean(mean_class) - 0.333) / np.std(std_class)

    df = pd.DataFrame(df)

    # Plotting
    ax = df.plot(kind='line', y='pval', x='nsub', ylim=(-0.01, 0.35),
            label='_nolegend_', linewidth=2, color='b', fontsize=12)
    ax.set_ylabel('Proportion of negative results', fontsize=15)
    ax.set_xlabel('Sample size', fontsize=15)
    ax.set_xticks(df['nsub'])
    ax.fill_between(df['nsub'], df['pval']-df['std'], df['pval']+df['std'], color='b', alpha=0.2)
    ax.axhline(y=0.01, linewidth=2, ls=':', color='black',
                label=r'$\alpha$ = 0.01')
    ax.legend(prop={'size': 20}, frameon=False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    to_save = opj(os.path.dirname(filepath), 'bootstrap_results.png')
    plt.savefig(to_save, dpi=200)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    project_dir = opj(home, 'Dropbox', 'ResMas_UvA', 'Thesis', 'Git')
    filepath = opj(project_dir, 'Analysis_results', 'benchmark_results.csv')

    main(filepath, 1000)

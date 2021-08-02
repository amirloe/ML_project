from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

import pandas as pd
import numpy as np


def friedman_test():
    """
    Collects the AUC for each algorithm and calculate friedman test. Returns the samples that ran through the test and
    statistic and p-val of the test
    """
    res = pd.read_excel('Results/Experiments_results.xlsx', sheet_name=None)
    baseline_auc = []
    paper_auc = []
    improve_auc = []
    for df_name, df in res.items():
        baseline_auc.append(df[df['Algorithm Name'] == 'Baseline']['AUC'].mean())
        paper_auc.append(df[df['Algorithm Name'] == 'Mean-Teacher']['AUC'].mean())
        improve_auc.append(df[df['Algorithm Name'] == 'Double-Mean-Teacher']['AUC'].mean())
    x = friedmanchisquare(baseline_auc, paper_auc, improve_auc)
    return baseline_auc, paper_auc, improve_auc, x


def post_hoc_test(baseline_auc, paper_auc, improve_auc):
    """
    Run posthoc_nemenyi test and print the results
    """
    data = np.array([baseline_auc, paper_auc, improve_auc])
    x = sp.posthoc_nemenyi_friedman(data.T)
    print(x)


def stat_test():
    baseline_auc, paper_auc, improve_auc, friedman = friedman_test()
    print(f'Friedman test p-value = {friedman.pvalue}')
    post_hoc_test(baseline_auc, paper_auc, improve_auc)

from numpy import result_type
import pandas as pd
from pathlib import Path


base_path = 'Tars_large/tars_ft_ml-false/'
n_shots = ['1', '2', '4', '8', '10', '50', '100', 'full']
reps = range(1, 6)


def get_series_from_repdir(n_shot, trial, name=None):
    with open(base_path + "/model_" + n_shot + '_shot/rep_' + trial + '/training.log', 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('macro avg'):
                l, p, r, f1, s = re.split("\s\s+", line)
    result_dict = {
        'n_shot': n_shot,
        'trial': trial,
        'precision': float(p) * 100,
        'recall': float(r) * 100,
        'f1-score': float(f1) * 100,
        'support':int(s)
    }
    return pd.DataFrame(result_dict, index=[n_shot + '_' + trial])


def create_tars_result_spreadsheets():
    all_results = pd.DataFrame()
    for n_shot in n_shots:
        for rep in reps:
            all_results = pd.concat([all_results, get_series_from_repdir(n_shot, str(rep))])
    all_results.to_csv("fs_results.csv")
    return all_results

import pdb
import re

if __name__ == '__main__':
    df = create_tars_result_spreadsheets()
    df_agg = df.groupby("n_shot").agg({
        'precision':['mean', 'std'],
        'recall':['mean', 'std'],
        'f1-score':['mean', 'std'],
    }).round(1)
    # import pdb; pdb.set_trace()
    print(df_agg.loc[n_shots])
    df_agg.loc[n_shots].to_csv("fs_results_agg.csv")

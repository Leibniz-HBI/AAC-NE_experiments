from numpy import result_type
import pandas as pd
from pathlib import Path


base_path = Path('Tars')


def get_series_from_repdir(repdir, name=None):
    with open(repdir/'training.log', 'r') as f:
        for line in f:
            if line.startswith('- F-score (micro) '):
                f_micro = float(line.strip().split('micro) ')[1])
            if line.startswith('- F-score (macro) '):
                f_macro = float(line.strip().split('macro) ')[1])
            if line.startswith('- Accuracy '):
                accuracy = float(line.strip().split('- Accuracy ')[1])
    result_dict = {
        'F-score (micro)': f_micro,
        'F-score (macro)': f_macro,
        'Accuracy': accuracy
        }
    return pd.Series(data=result_dict, index=result_dict.keys(), name=name)



def iter_shot_dirs(directory, shot):
    rep_data = []
    for rep_directory in directory.iterdir():
        if rep_directory.parts[-1].startswith('rep_') and rep_directory.is_dir():
            name = rep_directory.parts[-1]
            rep_data.append(get_series_from_repdir(rep_directory, name))
    df = pd.concat(rep_data, axis=1)
    df['mean'] = df.mean(axis=1)
    df = df.reindex(sorted(df.columns), axis=1)
    df.to_csv(directory/f'{shot}_shot_results.csv')
    return df['mean'].rename(shot)



def create_results(model):
    mean_shot_data = []
    for directory in base_path.iterdir():
        if model in str(directory) and directory.is_dir():
            shot = directory.parts[-1].split("_")[-2]
            mean_shot_data.append(iter_shot_dirs(directory, shot))
    df = pd.concat(mean_shot_data, axis=1)
    df = df.reindex(sorted(df.columns), axis=1)
    df.to_csv(base_path/f'{model}_results.csv')



def create_tars_result_spreadsheets():
    all_subs = [path.parts[-1] for path in base_path.iterdir()]
    models = {'_'.join(sub.split('_')[1:-2]) for sub in all_subs}
    for model in models:
        create_results(model)



if __name__ == '__main__':
    create_tars_result_spreadsheets()

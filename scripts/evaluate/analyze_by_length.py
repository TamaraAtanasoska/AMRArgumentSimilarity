import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.metrics import f1_score
from evaluate_dataset import mix


def get_binary_label(value, threshold):
    return 1 if value > threshold else 0


def get_scores_by_length(ds, thresholds, labels):
    ds['conclusion_standard_mixed'] = mix(ds['conclusion_standard'], ds['standard'])
    ds['conclusion_concept_mixed'] = mix(ds['conclusion_concept'], ds['standard'])
    ds['conclusion_structure_mixed'] = mix(ds['conclusion_structure'], ds['standard'])
    ds['summary_standard_mixed'] = mix(ds['summary_standard'], ds['standard'])
    ds['summary_concept_mixed'] = mix(ds['summary_concept'], ds['standard'])
    ds['summary_structure_mixed'] = mix(ds['summary_structure'], ds['standard'])

    for column, t in thresholds.items():
        ds[column + '_binary'] = ds[column].apply(lambda x: get_binary_label(x, t))

    data = {}
    for column, t in thresholds.items():
        data[column] = []
        for bin in labels:
            f1 = f1_score(ds[ds['bin'] == bin]['regression_label_binary'],
                          ds[ds['bin'] == bin][column + '_binary'], average="macro", zero_division=0)
            data[column].append(f1)

    df = pd.DataFrame(data, index=labels)
    df['mean'] = df.apply(np.mean, axis=1)
    df['count'] = ds['bin'].value_counts()
    df = df.transpose()
    return df


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path_preds_csv', type=lambda x: Path(x))
    arg_parser.add_argument('--data_path_res_csv', type=lambda x: Path(x))
    arg_parser.add_argument('--mixing_value', type=float, required=False, default=0.95)
    arg_parser.add_argument('--out_path', type=lambda x: Path(x))
    args = arg_parser.parse_args()

    df = pd.read_csv(args.data_path_preds_csv)
    res = pd.read_csv(args.data_path_res_csv, index_col=0)

    labels = ['<100', '100-200', '200-250', '250-300', '300-400', '400-500', '>500']

    by_len = get_scores_by_length(df, res['threshold'].astype(float), labels)
    by_len.to_csv(args.out_path / 'eval_by_length.csv')


if __name__ == '__main__':
    main()

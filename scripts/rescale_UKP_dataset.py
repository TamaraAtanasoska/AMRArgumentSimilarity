import argparse
from pathlib import Path

import pandas as pd


def read_UPK_data(path, sep='\t'):
    df = pd.read_csv(path, sep=sep)
    df = df[['topic', 'sentence_1', 'sentence_2', 'label']]
    df['regression_label_binary'] = df['label'].apply(lambda x: 1 if x in ['HS', 'SS'] else 0)
    return df[['topic', 'sentence_1', 'sentence_2', 'regression_label_binary']]


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--ukp_path', type=lambda x: Path(x))
    arg_parser.add_argument('--out_file', type=lambda x: Path(x))
    arg_parser.add_argument('--input_type', type=str, default='tsv')  # tsv or csv
    args = arg_parser.parse_args()

    sep = '\t' if args.input_type == 'tsv' else ','
    df = read_UPK_data(args.upk_path, sep=sep)
    df.to_csv(args.out_file, index=False)


if __name__ == "__main__":
    main()

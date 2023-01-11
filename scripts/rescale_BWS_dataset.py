import argparse
from pathlib import Path

import pandas as pd


def read_BWS_data(path):
    df = pd.read_csv(path)
    df = df[['argument1', 'argument2', 'topic', 'score']]
    df = df.rename(columns={'argument1': 'sentence_1', 'argument2': 'sentence_2', 'score': 'regression_label'})
    df['regression_label'] = df['regression_label'].apply(float)
    df['regression_label_binary'] = df['regression_label'].apply(lambda x: 1 if x > 0.5 else 0)
    return df[['topic', 'sentence_1', 'sentence_2', 'regression_label_binary', 'regression_label']]


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--bws_path', type=lambda x: Path(x))
    arg_parser.add_argument('--out_file', type=lambda x: Path(x))
    args = arg_parser.parse_args()

    df = read_BWS_data(args.bws_path)
    df.to_csv(args.out_file, index=False)


if __name__ == "__main__":
    main()

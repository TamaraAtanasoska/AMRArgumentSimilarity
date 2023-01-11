import argparse
from pathlib import Path

import pandas as pd


def min_max_normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)


def normalize_column(df, colname):
    min_value = df[colname].min()
    max_value = df[colname].max()
    return df[colname].apply(lambda x: min_max_normalize(x, min_value, max_value))


def correct_parsing_errors(text):
    text = text.replace('‰Ыќ', '"')
    text = text.replace('‰ЫП', '"')
    text = text.replace('‰ЫЄ', "'")
    text = text.replace('‰ЫТ', ": ")
    text = text.replace('‰ЫУ', '"')
    text = text.replace('‰Ыў', '')
    text = text.replace('‰Ы_', '')
    text = text.replace('  ', ' ').replace('  ', ' ')
    assert '‰' not in text, text
    return text


def read_AFS_data(path, name, encoding='cp1251'):
    df = pd.read_csv(path / f'ArgPairs_{name}.csv', encoding=encoding)
    df = df[['regression_label', 'sentence_1', 'sentence_2']]
    df['sentence_1'] = df['sentence_1'].apply(correct_parsing_errors)
    df['sentence_2'] = df['sentence_2'].apply(correct_parsing_errors)
    df['topic'] = name
    df['regression_label'] = df['regression_label'].apply(float)
    df['regression_label_binary'] = df['regression_label'].apply(lambda x: 1 if x > 3 else 0)
    df['regression_label_scaled'] = normalize_column(df, 'regression_label')
    return df[['topic', 'sentence_1', 'sentence_2', 'regression_label_binary', 'regression_label']]


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--afs_path', type=lambda x: Path(x))
    arg_parser.add_argument('--out_file', type=lambda x: Path(x))
    arg_parser.add_argument('--encoding', required=False, default='cp1251')  # the dataset is distributed in cp1251
    args = arg_parser.parse_args()

    df_dp = read_AFS_data(args.afs_path, 'DP', encoding=args.encoding)
    df_gm = read_AFS_data(args.afs_path, 'GM', encoding=args.encoding)
    df_gc = read_AFS_data(args.afs_path, 'GC', encoding=args.encoding)
    df = pd.concat([df_dp, df_gm, df_gc])
    df.to_csv(args.out_file, index=False)  # the combined dataset is saved in default utf-8


if __name__ == "__main__":
    main()

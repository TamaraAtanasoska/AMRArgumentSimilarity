import argparse
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

def read_smatch_scores(f):
    scores = []
    with open(f, "r") as f:
        lines = f.read().split("\n")
    for line in lines[:-1]:
        if line.startswith('Smatch score F1'):
            scores.append(float(line.replace("Smatch score F1 ", "")))
    return scores


def add_smatch_scores(df, scores, col_name):
    assert len(scores) == len(df), f'{col_name} expected scores of length {len(df)} got {len(scores)}'
    df[col_name] = scores


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path_csv', type=lambda x: Path(x))
    arg_parser.add_argument('--data_path_smatch_standard', type=lambda x: Path(x))
    arg_parser.add_argument('--data_path_smatch_struct', type=lambda x: Path(x), required=False, default=None)
    arg_parser.add_argument('--data_path_smatch_concept', type=lambda x: Path(x), required=False, default=None)
    arg_parser.add_argument('--data_path_smatch_conclusion_standard', type=lambda x: Path(x), required=False,
                            default=None)
    arg_parser.add_argument('--data_path_smatch_conclusion_struct', type=lambda x: Path(x), required=False,
                            default=None)
    arg_parser.add_argument('--data_path_smatch_conclusion_concept', type=lambda x: Path(x), required=False,
                            default=None)
    arg_parser.add_argument('--data_path_smatch_summary_standard', type=lambda x: Path(x), required=False, default=None)
    arg_parser.add_argument('--data_path_smatch_summary_struct', type=lambda x: Path(x), required=False, default=None)
    arg_parser.add_argument('--data_path_smatch_summary_concept', type=lambda x: Path(x), required=False, default=None)
    arg_parser.add_argument('--column_name1', type=str, required=False, default='sentence_1')
    arg_parser.add_argument('--column_name2', type=str, required=False, default='sentence_2')
    arg_parser.add_argument('--calculate_length', required=False, default=True)
    arg_parser.add_argument('--out_path', type=lambda x: Path(x))
    args = arg_parser.parse_args()

    df = pd.read_csv(args.data_path_csv)

    add_smatch_scores(df, read_smatch_scores(args.data_path_smatch_standard), 'standard')
    if args.data_path_smatch_struct:
        add_smatch_scores(df, read_smatch_scores(args.data_path_smatch_struct), 'structure')
    if args.data_path_smatch_concept:
        add_smatch_scores(df, read_smatch_scores(args.data_path_smatch_concept), 'concept')

    if args.data_path_smatch_conclusion_standard:
        add_smatch_scores(df, read_smatch_scores(args.data_path_smatch_conclusion_standard), 'conclusion_standard')
    if args.data_path_smatch_conclusion_struct:
        add_smatch_scores(df, read_smatch_scores(args.data_path_smatch_conclusion_struct), 'conclusion_structure')
    if args.data_path_smatch_conclusion_concept:
        add_smatch_scores(df, read_smatch_scores(args.data_path_smatch_conclusion_concept), 'conclusion_concept')

    if args.data_path_smatch_summary_standard:
        add_smatch_scores(df, read_smatch_scores(args.data_path_smatch_summary_standard), 'summary_standard')
    if args.data_path_smatch_summary_struct:
        add_smatch_scores(df, read_smatch_scores(args.data_path_smatch_summary_struct), 'summary_structure')
    if args.data_path_smatch_summary_concept:
        add_smatch_scores(df, read_smatch_scores(args.data_path_smatch_summary_concept), 'summary_concept')

    if args.calculate_length and args.calculate_length not in ['F', 'False', '0']:
        df['sentence_1_len'] = df[args.column_name1].apply(len)
        df['sentence_2_len'] = df[args.column_name2].apply(len)
        df['combined_len'] = df['sentence_1_len'] + df['sentence_2_len']
        bins = [0, 100, 200, 250, 300, 400, 500, 1000]
        labels = ['<100', '100-200', '200-250', '250-300', '300-400', '400-500', '>500']
        df['bin'] = pd.cut(x=df['combined_len'], bins=bins, labels=labels, include_lowest=True)

    df.to_csv(args.out_path / 'df_smatch_scores.csv', index=False)


if __name__ == '__main__':
    main()

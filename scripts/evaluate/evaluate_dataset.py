import argparse
import pandas as pd

import crval

from pathlib import Path
from scipy.stats import spearmanr


def mix(val1, val2, mixing_value=0.95):
    assert len(val1) == len(val2)
    return (1 - mixing_value) * val1 + mixing_value * val2


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path_preds_csv', type=lambda x: Path(x))
    arg_parser.add_argument('--fold_size', type=int)
    arg_parser.add_argument('--mixing_value', type=float, required=False, default=0.95)
    arg_parser.add_argument('--correlation_column', type=str, required=False, default=None)
    args = arg_parser.parse_args()

    df = pd.read_csv(args.data_path_preds_csv)

    true = df['regression_label_binary'].to_numpy()
    topics = df['topic'].to_numpy()

    preds_st = df['standard'].astype(float).to_numpy()
    preds_c = df['conclusion_concept'].astype(float).to_numpy()
    preds_s = df['structure'].astype(float).to_numpy()

    preds_co_st = df['conclusion_standard'].astype(float).to_numpy()
    preds_co_c = df['conclusion_concept'].astype(float).to_numpy()
    preds_co_s = df['conclusion_structure'].to_numpy()

    preds_con_st = mix(preds_co_st, preds_st, args.mixing_value)
    preds_con_c = mix(preds_co_c, preds_c, args.mixing_value)
    preds_con_s = mix(preds_co_s, preds_s, args.mixing_value)

    preds_so_st = df['summary_standard'].astype(float).to_numpy()
    preds_so_c = df['summary_concept'].astype(float).to_numpy()
    preds_so_s = df['summary_structure'].astype(float).to_numpy()

    preds_sum_st = mix(preds_so_st, preds_st, args.mixing_value)
    preds_sum_c = mix(preds_so_c, preds_c, args.mixing_value)
    preds_sum_s = mix(preds_so_s, preds_s, args.mixing_value)

    print('____standard____')
    crval.runcv(preds_st, true, topics, fold_size=args.fold_size)
    print('____concept____')
    crval.runcv(preds_c, true, topics, fold_size=args.fold_size)
    print('____structure____')
    crval.runcv(preds_s, true, topics, fold_size=args.fold_size)

    print('____conclusion_standard____')
    crval.runcv(preds_con_st, true, topics, fold_size=args.fold_size)
    print('____conclusion_concept____')
    crval.runcv(preds_con_c, true, topics, fold_size=args.fold_size)
    print('____conclusion_structure____')
    crval.runcv(preds_con_s, true, topics, fold_size=args.fold_size)

    print('____conclusion_only_standard____')
    crval.runcv(preds_co_st, true, topics, fold_size=args.fold_size)
    print('____conclusion_only_concept____')
    crval.runcv(preds_co_c, true, topics, fold_size=args.fold_size)
    print('____conclusion_only_structure____')
    crval.runcv(preds_co_s, true, topics, fold_size=args.fold_size)

    print('____summary_standard____')
    crval.runcv(preds_sum_st, true, topics, fold_size=args.fold_size)
    print('____summary_concept____')
    crval.runcv(preds_sum_c, true, topics, fold_size=args.fold_size)
    print('____summary_structure____')
    crval.runcv(preds_sum_s, true, topics, fold_size=args.fold_size)

    print('____summary_only_standard____')
    crval.runcv(preds_so_st, true, topics, fold_size=args.fold_size)
    print('____summary_only_concept____')
    crval.runcv(preds_so_c, true, topics, fold_size=args.fold_size)
    print('____summary_only_structure____')
    crval.runcv(preds_so_s, true, topics, fold_size=args.fold_size)

    if args.correlation_column:
        correlation_tgt = df[args.correlation_column].astype(float).to_numpy()
        print('____correlations____')

        print('____standard____')
        print(spearmanr(preds_st, correlation_tgt))
        print('____concept____')
        print(spearmanr(preds_c, correlation_tgt))
        print('____structure____')
        print(spearmanr(preds_s, correlation_tgt))

        print('____conclusion_standard____')
        print(spearmanr(preds_con_st, correlation_tgt))
        print('____conclusion_concept____')
        print(spearmanr(preds_con_c, correlation_tgt))
        print('____conclusion_structure____')
        print(spearmanr(preds_con_s, correlation_tgt))

        print('____conclusion_only_standard____')
        print(spearmanr(preds_co_st, correlation_tgt))
        print('____conclusion_only_concept____')
        print(spearmanr(preds_co_c, correlation_tgt))
        print('____conclusion_only_structure____')
        print(spearmanr(preds_co_s, correlation_tgt))

        print('____summary_standard____')
        print(spearmanr(preds_sum_st, correlation_tgt))
        print('____summary_concept____')
        print(spearmanr(preds_sum_c, correlation_tgt))
        print('____summary_structure____')
        print(spearmanr(preds_sum_s, correlation_tgt))

        print('____summary_only_standard____')
        print(spearmanr(preds_so_st, correlation_tgt))
        print('____summary_only_concept____')
        print(spearmanr(preds_so_c, correlation_tgt))
        print('____summary_only_structure____')
        print(spearmanr(preds_so_s, correlation_tgt))


if __name__ == '__main__':
    main()

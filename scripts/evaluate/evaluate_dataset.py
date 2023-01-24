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
    arg_parser.add_argument('--out_path', type=lambda x: Path(x))
    args = arg_parser.parse_args()

    df = pd.read_csv(args.data_path_preds_csv)

    true = df['regression_label_binary'].to_numpy()
    topics = df['topic'].to_numpy()

    preds = {}
    columns = ['standard', 'concept', 'structure',
               'conclusion_standard', 'conclusion_concept', 'conclusion_structure',
               'summary_standard', 'summary_concept', 'summary_structure']
    for col in columns:
        preds[col] = df[col].astype(float).to_numpy()

    preds['conclusion_standard_mixed'] = mix(preds['conclusion_standard'], preds['standard'], args.mixing_value)
    preds['conclusion_concept_mixed'] = mix(preds['conclusion_concept'], preds['concept'], args.mixing_value)
    preds['conclusion_structure_mixed'] = mix(preds['conclusion_structure'], preds['structure'], args.mixing_value)

    preds['summary_standard_mixed'] = mix(preds['summary_standard'], preds['standard'], args.mixing_value)
    preds['summary_concept_mixed'] = mix(preds['summary_concept'], preds['concept'], args.mixing_value)
    preds['summary_structure_mixed'] = mix(preds['summary_structure'], preds['structure'], args.mixing_value)

    res = {}
    for col in preds.keys():
        res[col] = {}
        print(f'____{col}____')
        eval_score, _, _, threshold = crval.runcv(preds[col], true, topics, fold_size=args.fold_size)
        res[col]['f1'] = eval_score
        res[col]['threshold'] = threshold
        if args.correlation_column:
            correlation_tgt = df[args.correlation_column].astype(float).to_numpy()
            print('__correlation__')
            print(spearmanr(preds[col], correlation_tgt))
            res[col]['correlation'], res[col]['correlation_p'] = spearmanr(preds[col], correlation_tgt)

    pd.DataFrame(res).transpose().to_csv(args.out_path / 'results.csv')


if __name__ == '__main__':
    main()

import amrlib
import argparse
import csv
from tqdm import tqdm
from pathlib import Path


def read_csv_columns(path_to_csv, column_name1, column_name2):
    source_column, target_column = [], []
    with open(path_to_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_column.append(row[column_name1])
            target_column.append(row[column_name2])
    return source_column, target_column


def process_in_batches(function, array, batch_size):
    results = []
    array_batches = [array[i:i + batch_size] for i in range(0, len(array), batch_size)]
    for batch in tqdm(array_batches):
        results += function(batch)
    return results


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path_csv', type=lambda x: Path(x))
    arg_parser.add_argument('--column_name1', type=str, default='sentence_1')
    arg_parser.add_argument('--column_name2', type=str, default='sentence_2')
    arg_parser.add_argument('--batch_size', type=int, default=5)
    arg_parser.add_argument('--out_path', type=lambda x: Path(x))
    args = arg_parser.parse_args()

    source_column, target_column = read_csv_columns(args.data_path_csv, args.column_name1, args.column_name2)

    stog = amrlib.load_stog_model()
    source_graphs = process_in_batches(function=stog.parse_sents, array=source_column, batch_size=args.batch_size)

    with open(args.out_path / 'amr.src', 'w') as f:
        for graph in source_graphs:
            f.write(graph)
            f.write('\n')

    target_graphs = process_in_batches(function=stog.parse_sents, array=target_column, batch_size=args.batch_size)
    with open(args.out_path / 'amr.tgt', 'w') as f:
        for graph in target_graphs:
            f.write(graph)
            f.write('\n')


if __name__ == '__main__':
    main()

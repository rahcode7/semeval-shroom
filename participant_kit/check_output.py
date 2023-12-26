# command: python3 $program/score.py $input/res $input/ref $output
# description: run scoring program on any data

import argparse
import collections
import json
import pathlib

from scipy.stats import spearmanr

parser = argparse.ArgumentParser()
parser.add_argument('submission', type=pathlib.Path)
parser.add_argument('--is_val', action='store_true')
args = parser.parse_args()

submitted_files = list(args.submission.glob('**/*.json'))
n_ref_files = 2
assert 0 < len(submitted_files) <= n_ref_files, \
    f'Expected 0 < n files <= {n_ref_files}, got {len(submitted_files)}'

prefix = 'val' if args.is_val else 'test'

for sub_file in submitted_files:

    assert sub_file.name in (f'{prefix}.model-aware.json', f'{prefix}.model-agnostic.json'), \
        f'Invalid file name in submission `{sub_file.name}`'

    with open(sub_file, 'r') as istr:
        sub_data = json.load(istr)
        if not args.is_val:
            sub_data = sorted(sub_data, key=lambda item: item['id'])
            
    num_items = 1500 if not args.is_val else 501 if 'aware' in sub_file.name else 499
    assert len(sub_data) == num_items, \
        f'Invalid number of items in "{sub_file.name}": ' \
        f'expected {num_items}, got {len(sub_data)}'

    expected_keys = {'label', 'p(Hallucination)'}
    if not args.is_val:
        expected_keys.add('id')
    assert all(
        (set(sub.keys()) & expected_keys) == expected_keys
        for sub in sub_data
    ), f'Not all datapoints contain all expected keys'

    assert all(
        sub['label'] in {'Hallucination', 'Not Hallucination'} 
        for sub in sub_data
    ), 'not all labels have the correct format'


print('all clear!')

"""

Use once to figure out the number of training examples

"""

import os
import json
import glob

from tqdm import tqdm

DATA_DIR = '/disk3/realnews-merged-batches/'
LAST_EPOCH = 1200

import re

fnames = sorted(glob.glob(f'{DATA_DIR}/*'), key=lambda s: int(re.match(r'.*epoch_([0-9]+).*', s).group(1)))
epoch_files = []
metric_files = []
for fname in fnames:
    if fname.endswith('_metrics.json'):
        metric_files.append(os.path.join(DATA_DIR, fname))
    else:
        epoch_files.append(os.path.join(DATA_DIR, fname))
    if re.match(r'.*epoch_([0-9]+).*', fname).group(1) == LAST_EPOCH:
        break
assert len(epoch_files) == len(metric_files)

num_training_examples_per_epoch = []
for metric_file in tqdm(metric_files):
    with open(metric_file) as f_metric:
        for line in f_metric:
            num_training_examples_per_epoch.append(json.loads(line)['num_training_examples'])

print(f'Total number of training examples: {sum(num_training_examples_per_epoch)}')


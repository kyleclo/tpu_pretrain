"""

Use once to figure out the number of training examples

"""

import os
import json

from tqdm import tqdm

DATA_DIR = '/disk2/med-merged/'

fnames = sorted(os.listdir(DATA_DIR))
epoch_files = []
metric_files = []
for fname in fnames:
    if fname.endswith('_metrics.json'):
        metric_files.append(os.path.join(DATA_DIR, fname))
    else:
        epoch_files.append(os.path.join(DATA_DIR, fname))
assert len(epoch_files) == len(metric_files)

num_training_examples_per_epoch = []
for metric_file in tqdm(metric_files):
    with open(metric_file) as f_metric:
        for line in f_metric:
            num_training_examples_per_epoch.append(json.loads(line)['num_training_examples'])

print(f'Total number of training examples: {sum(num_training_examples_per_epoch)}')


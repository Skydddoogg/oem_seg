import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import ast
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model1')
parser.add_argument('--model2')
parser.add_argument('--title')

args = parser.parse_args()

d = {'epoch': [], 'normalized loss': [], '': [], 'method': []}
with open(os.path.join('outputs', f'{args.model1}_train_loss.txt')) as f:
    loss_list = ast.literal_eval(f.readlines()[0])
    d['epoch'] += [i for i in range(len(loss_list))]
    d['normalized loss'] += loss_list
    d[''] += ['train'] * len(loss_list)
    d['method'] += ['w/o SMOD'] * len(loss_list)

with open(os.path.join('outputs', f'{args.model1}_val_loss.txt')) as f:
    loss_list = ast.literal_eval(f.readlines()[0])
    d['epoch'] += [i for i in range(len(loss_list))]
    d['normalized loss'] += loss_list
    d[''] += ['val'] * len(loss_list)
    d['method'] += ['w/o SMOD'] * len(loss_list)

d['normalized loss'] = np.array(d['normalized loss'])
d['normalized loss'] = (d['normalized loss'] - min(d['normalized loss'])) / (max(d['normalized loss']) - min(d['normalized loss']))

d2 = {'epoch': [], 'normalized loss': [], '': [], 'method': []}
with open(os.path.join('outputs', f'{args.model2}_train_loss.txt')) as f:
    loss_list = ast.literal_eval(f.readlines()[0])
    d2['epoch'] += [i for i in range(len(loss_list))]
    d2['normalized loss'] += loss_list
    d2[''] += ['train'] * len(loss_list)
    d2['method'] += ['w/ SMOD'] * len(loss_list)

with open(os.path.join('outputs', f'{args.model2}_val_loss.txt')) as f:
    loss_list = ast.literal_eval(f.readlines()[0])
    d2['epoch'] += [i for i in range(len(loss_list))]
    d2['normalized loss'] += loss_list
    d2[''] += ['val'] * len(loss_list)
    d2['method'] += ['w/ SMOD'] * len(loss_list)

d2['normalized loss'] = np.array(d2['normalized loss'])
d2['normalized loss'] = (d2['normalized loss'] - min(d2['normalized loss'])) / (max(d2['normalized loss']) - min(d2['normalized loss']))

d['epoch'] += d2['epoch']
d['normalized loss'] = list(d['normalized loss']) + list(d2['normalized loss'])
d[''] += d2['']
d['method'] += d2['method']

df = pd.DataFrame(data=d)

g = sns.FacetGrid(df, col="method", hue='')
g.map(sns.lineplot, 'epoch', 'normalized loss')
g.add_legend()
g.tight_layout()
g.savefig(os.path.join('outputs', f'{args.model1}_vs_{args.model2}.png'), dpi=300)

# if args.title is not None:
#     plt.title(f'{args.title}') # Edit here
# plt.xlabel('epoch')
# plt.ylabel('Normalized loss')
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.show()
# plt.savefig(os.path.join('outputs', f'{args.model}.png'), dpi=300) # Edit here
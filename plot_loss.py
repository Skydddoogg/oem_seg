import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import ast
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model')
parser.add_argument('--title')

args = parser.parse_args()

d = {'x': [], 'y': [], 'label': []}
with open(os.path.join('outputs', f'{args.model}_train_loss.txt')) as f:
    loss_list = ast.literal_eval(f.readlines()[0])
    d['x'] += [i for i in range(len(loss_list))]
    d['y'] += loss_list
    d['label'] += ['train'] * len(loss_list)

with open(os.path.join('outputs', f'{args.model}_val_loss.txt')) as f:
    loss_list = ast.literal_eval(f.readlines()[0])
    d['x'] += [i for i in range(len(loss_list))]
    d['y'] += loss_list
    d['label'] += ['val'] * len(loss_list)

d['y'] = np.array(d['y'])
d['y'] = (d['y'] - min(d['y'])) / (max(d['y']) - min(d['y']))

df = pd.DataFrame(data=d)

sns.lineplot(
        data=df, 
        x="x", y="y", 
        hue='label', 
        style="label"
)

if args.title is not None:
    plt.title(f'{args.title}') # Edit here
plt.xlabel('Epoch')
plt.ylabel('Normalized loss')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join('outputs', f'{args.model}.png'), dpi=300) # Edit here
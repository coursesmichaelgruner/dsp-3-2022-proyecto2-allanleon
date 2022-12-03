#!/usr/bin/env python3

import sys
import os
import glob
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print(f'usage {sys.argv[0]} ./path_to_root_of_data')
    sys.exit(1)

path = sys.argv[1]

if not os.path.isdir(path):
    print('path does not exist')
    sys.exit(1)


training = os.path.join(path,'training')
validation = os.path.join(path,'validation')

types = ['training','validation']
types_title= ['Training labels distribution','validation labels distribution']

cmds = [
    'down',
    'go',
    'left',
    'no',
    'off',
    'on',
    'right',
    'stop',
    'unknown',
    'up',
    'yes',
    'background_noise'
]

cmd_show=cmds.copy()
cmd_show[-1]='background noise'

fig,ax = plt.subplots(nrows=2,ncols=1)

for i in range(len(types)):
    count = []
    for cmd in cmds:
        audio_file = glob.glob(f'{path}/{types[i]}/{cmd}/*.npy')
        count.append(len(audio_file))
    ax[i].bar(cmd_show,count)
    ax[i].set_title(types[i])
    plt.setp(ax[i].get_xticklabels(), rotation=30, horizontalalignment='right')

fig.tight_layout(pad=1.0)
fig.savefig('img/label_dist.pdf')
plt.show()


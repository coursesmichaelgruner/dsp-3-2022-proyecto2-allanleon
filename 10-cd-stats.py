#!/usr/bin/python3

import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas

if len(sys.argv) != 2:
    print(f'usage {sys.argv[0]}  <log of trained batches "e.g: training_batches.log">')
    sys.exit(1)

file = sys.argv[1]

if not os.path.exists(file):
    print(f'file "{file}" does not exist')
    sys.exit(1)

stats = pandas.read_csv(file,header=0)


fig, axes = plt.subplots(nrows=2, ncols=1)

for ax in range(2):
    values = stats.values[:,0::(ax+1)]
    axes[ax].plot(values[:,0],values[:,1])
    axes[ax].set_title(stats.columns[ax+1])

    epochs = stats.values[::169,0::(ax+1)]

    axes[ax].plot(epochs[:,0],epochs[:,1],marker=matplotlib.markers.CARETDOWNBASE,linestyle='')
    axes[ax].legend(['Batch','Epoch'])

fig.tight_layout(pad=1)
fig.savefig('img/training_stats.pdf')
plt.show()
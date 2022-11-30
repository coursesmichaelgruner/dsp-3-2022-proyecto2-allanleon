#!/usr/bin/python3

import tensorflow as tf

from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
import os
import re

rows = 40
cols = 100

if len(sys.argv) != 2:
    print('usage 7a-normailze.py ./path_to_root_of_data')
    sys.exit(1)

path = sys.argv[1]

labels = ['down', 'go', 'left', 'no', 'off',
          'on', 'right', 'stop', 'unknown', 'up', 'yes', 'background_noise']


validation_list_f = open(path+'/validation_list.txt', 'r')
validation_list = validation_list_f.readlines()
validation_list_f.close()
validation_list = [f'{path}/{x}'.strip() for x in validation_list]


def get_label(file: str):
    for i in range(len(labels)):
        label = labels[i]
        if re.search(f'/{label}/.*.npy', file) is not None:
            return i


def normalize(data):
    mean = np.mean(data)
    var = np.var(data)
    if (abs(var) < sys.float_info.epsilon):
        return np.zeros(data.shape)
    data = (data - mean)/np.sqrt(var)
    return data


def get_array(file_list: list[str], rows=40, cols=100):
    arr = np.empty((len(file_list), rows, cols))
    labels = np.empty(len(file_list), dtype=np.uint8)
    for i in range(len(file_list)):
        file = file_list[i]
        if not file:
            continue
        file = file.replace('.wav', '.npy')

        if not os.path.isfile(file):
            print(f'{file} not found')
            sys.exit(1)
        label = get_label(file)

        data = np.load(file)
        r, c = data.shape
        if r != rows or c != cols:
            data = np.pad(data, [(0, rows-r), (0, cols-c)],
                          mode='constant', constant_values=0)

        arr[i] = normalize(data)
        labels[i] = label
    return (arr, labels)


print('preparing data')

validation_x,labels_valid=get_array(validation_list,rows,cols)
print('preparing data done')
print(validation_x[0:1].shape)
model=keras.models.load_model('model.h5')

results = model.predict(validation_x[-1:])
print(results)


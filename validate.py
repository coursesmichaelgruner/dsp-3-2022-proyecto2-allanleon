#!/usr/bin/python3

import tensorflow as tf

from tensorflow import keras
import matplotlib.pyplot as plt
from utils import get_array
import sys


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

print('preparing data')

validation_x, labels_valid = get_array(validation_list, labels, rows, cols)
print('preparing data done')
print(validation_x[0:1].shape)
model = keras.models.load_model('model.h5')

results = model.predict(validation_x[-1:])
print(results)

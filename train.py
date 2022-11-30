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

training_list_f = open(path+'/training_list.txt', 'r')
training_list = training_list_f.readlines()
training_list_f.close()
training_list = [f'{path}/{x}'.strip() for x in training_list]

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
training_x, training_labels = get_array(training_list, rows, cols)
# validation_x,labels_valid=get_array(validation_list,rows,cols)
print('preparing data done')
print(training_x.shape)

data_format = "channels_last"

inp_shape = (rows, cols, 1)
model = keras.models.Sequential()

# 2. Convolution2d
model.add(keras.layers.Conv2D(filters=12, kernel_size=(3, 3),
          input_shape=inp_shape, data_format=data_format, padding='same'))

# 3. Batch Normalization
model.add(keras.layers.BatchNormalization())

# 4. ReLU
model.add(keras.layers.Activation(keras.activations.relu))

# 5. MaxPooling2D
model.add(keras.layers.MaxPooling2D(
    pool_size=(3, 3), strides=(2, 2), padding='same'))

# 6. Convolution2d
model.add(keras.layers.Conv2D(filters=24, kernel_size=(3, 3), padding='same'))

# 7. Batch Normalization
model.add(keras.layers.BatchNormalization())

# 8. ReLU
model.add(keras.layers.Activation(keras.activations.relu))

# 9. MaxPooling2D
model.add(keras.layers.MaxPooling2D(
    pool_size=(3, 3), strides=(2, 2), padding='same'))

# 10. Convolution2d
model.add(keras.layers.Conv2D(filters=48, kernel_size=(3, 3), padding='same'))

# 11. Batch Normalization
model.add(keras.layers.BatchNormalization())

# 12. ReLU
model.add(keras.layers.Activation(keras.activations.relu))

# 13. MaxPooling2D
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3),
          strides=(2, 2), padding='same'))

# 14. Convolution2d
model.add(keras.layers.Conv2D(filters=48, kernel_size=(3, 3), padding='same'))

# 15. Batch Normalization
model.add(keras.layers.BatchNormalization())

# 16. ReLU
model.add(keras.layers.Activation(keras.activations.relu))

# 17. Convolution2d
model.add(keras.layers.Conv2D(filters=48, kernel_size=(3, 3), padding='same'))

# 18. Batch Normalization
model.add(keras.layers.BatchNormalization())

# 19. ReLU
model.add(keras.layers.Activation(keras.activations.relu))

# 20. MaxPooling2D
model.add(keras.layers.MaxPooling2D(
    pool_size=(1, 13), strides=(1, 1), padding='same'))

# 21. Dropout
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Flatten())

# 22. Dense
model.add(keras.layers.Dense(12, activation='softmax'))

# 23. Softmax
#model.add(keras.layers.Softmax())

model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4),
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=[keras.metrics.CategoricalAccuracy(),
                       keras.metrics.FalseNegatives()])

model.fit(x=training_x, y=keras.utils.to_categorical(training_labels), batch_size=128, epochs=25, shuffle=True)

model.save('model.h5')

print('hola')

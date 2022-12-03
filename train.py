#!/usr/bin/python3

import tensorflow as tf

from tensorflow import keras
import sys
from utils import get_array

rows = 40
cols = 100

if len(sys.argv) != 2:
    print('usage ./train.py ./path_to_root_of_data')
    sys.exit(1)

path = sys.argv[1]

labels = ['down', 'go', 'left', 'no', 'off',
          'on', 'right', 'stop', 'unknown', 'up', 'yes', 'background_noise']

training_list_f = open(path+'/training_list.txt', 'r')
training_list = training_list_f.readlines()
training_list_f.close()
training_list = [f'{path}/{x}'.strip() for x in training_list]


print('preparing data')
training_x, training_labels = get_array(training_list, labels, rows, cols)

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
# model.add(keras.layers.Softmax())

model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4),
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=[keras.metrics.CategoricalAccuracy(),
                       keras.metrics.FalseNegatives()])

class LogBatchCallback(keras.callbacks.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.global_batch = 0
        self.filename='training_batches.log'
        with open(self.filename,'w'):
            pass

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())

        with open(self.filename,'a') as f:
            if self.global_batch==0:
                f.write(f'batch,{keys[0]},{keys[1]}\n')
            f.write(f'{self.global_batch},{logs[keys[0]]},{logs[keys[1]]}\n')

        self.global_batch += 1

log_batch_callback = LogBatchCallback()

model.fit(x=training_x, y=keras.utils.to_categorical(training_labels),
          batch_size=128, epochs=25, shuffle=True, callbacks=[log_batch_callback])

model.save('model.h5')

print('hola')

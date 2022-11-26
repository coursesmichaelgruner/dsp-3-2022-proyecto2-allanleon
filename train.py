#!/usr/bin/python3

import tensorflow as tf

from tensorflow import keras
import matplotlib.pyplot as plt

data_format = "channels_last"

inp_shape = (40, 100, 1)
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
    pool_size=(1, 3), strides=(1, 1), padding='same'))

# 21. Dropout
model.add(keras.layers.Dropout(0.2))

# 22. Dense
model.add(keras.layers.Dense(12))

# 23. Softmax
model.add(keras.layers.Softmax())

model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4),
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=[keras.metrics.CategoricalAccuracy(),
                       keras.metrics.FalseNegatives()])

model.fit(x=None,y=None,batch_size=128,epochs=25,shuffle=True)

model.save('model.h5')

print('hola')

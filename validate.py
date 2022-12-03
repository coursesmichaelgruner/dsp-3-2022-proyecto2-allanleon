#!/usr/bin/python3

from tensorflow import keras
import numpy as np
from utils import get_array
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model

rows = 40
cols = 100

if len(sys.argv) != 2:
    print('usage ./validate.py ./path_to_root_of_data')
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

plot_model(model, to_file='img/model_plot.pdf', show_shapes=True, show_layer_names=True)

results = model.predict(validation_x)
results = np.argmax(results,axis=1)

labels[-1]='noise'
confusion = confusion_matrix(labels_valid, results , normalize='pred')

disp=ConfusionMatrixDisplay(confusion,display_labels=labels)

fig, ax = plt.subplots(figsize=(10,10))

disp.plot(ax=ax,values_format='.2f')

plt.xlabel("Predicted")
plt.ylabel("Real class")
plt.savefig('img/confusion.pdf')
plt.show()
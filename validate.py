#!/usr/bin/python3
from oct2py import octave
from tensorflow import keras
import numpy as np
from utils import get_array
import sys
from keras.utils.vis_utils import plot_model
import time
import multiprocessing as mp
import psutil


def fn(n=5):
    global results
    global labels
    global labels_valid
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

    glob_ts=0
    for i in range(n):
        ts= time.time()
        results = model.predict(validation_x)
        ts= time.time()-ts
        glob_ts +=ts

    print(f'{ts:.3f} s: {len(validation_list)/ts:.0f} spectogram/s')
    print(f'{ts:.3f} s: {ts/len(validation_list)*1000:.3f} ms/spectogram')


def monitor(target):
    worker_process = mp.Process(target=target)
    worker_process.start()
    p = psutil.Process(worker_process.pid)
    cpu_percents=[]
    while worker_process.is_alive():
        cpu_percents.append(p.cpu_percent(1))

    worker_process.join()
    return cpu_percents

cpu_percent = monitor(target=fn)
print(np.max(np.array(cpu_percent)/psutil.cpu_count()*2))
fn(1)
results = np.argmax(results,axis=1)
print(results)

Yt = labels_valid
Yp = results
octave.run('confusion.m')
labels[-1]='noise'
octave.confusion_matrix(Yt,Yp,labels)
input('Press enter to close')
#!/usr/bin/python3

from tensorflow import keras
import numpy as np
from utils import get_spectograms
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from spectrogram import compute_spectrogram
import librosa
import os
import glob


if len(sys.argv) != 2:
    print('usage ./predict.py audio_file|directory')
    sys.exit(1)

path = sys.argv[1]

if os.path.isdir(path):
    files = glob.glob(f'{path}/*.wav')
elif os.path.isfile(path):
    files = [path]
else:
    print('path not found')
    sys.exit(1)

labels = ['down', 'go', 'left', 'no', 'off',
          'on', 'right', 'stop', 'unknown', 'up', 'yes', 'background_noise']


spectrograms = get_spectograms(files)

model = keras.models.load_model('model.h5')

results = model.predict(spectrograms)

indx = np.argmax(results, axis=1)

print('==============\n')

for i in range(len(files)):
    print(f'{os.path.basename(files[i])}: \t{labels[indx[i]]} \t{results[i][indx[i]]*100:.3f}%')

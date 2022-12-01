#!/usr/bin/python3

from tensorflow import keras
import numpy as np
from utils import normalize
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from spectrogram import compute_spectrogram
import librosa


if len(sys.argv) != 2:
    print('usage ./predict.py audio_file')
    sys.exit(1)

path = sys.argv[1]

labels = ['down', 'go', 'left', 'no', 'off',
          'on', 'right', 'stop', 'unknown', 'up', 'yes', 'background_noise']

y, fs = librosa.load(path, sr=16000)

spectrogram = compute_spectrogram(y,fs)
spectrogram=normalize(spectrogram)
spectrogram = spectrogram.reshape((1,spectrogram.shape[0],spectrogram.shape[1]))

model = keras.models.load_model('model.h5')

results = model.predict(spectrogram)

indx = np.argmax(results,axis=1)[0]

print('==============\n')

print(f'{labels[indx]} {results[0][indx]*100:.2f}')

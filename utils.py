import numpy as np
import sys
import os
import re
from spectrogram import compute_spectrogram
import librosa


def get_label(file: str, labels: list[str]):
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


def get_array(file_list: list[str], label_names: list[str], rows=40, cols=100):
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
        label = get_label(file, label_names)

        data = np.load(file)
        r, c = data.shape
        if r != rows or c != cols:
            data = np.pad(data, [(0, rows-r), (0, cols-c)],
                          mode='constant', constant_values=0)

        arr[i] = normalize(data)
        labels[i] = label
    return (arr, labels)


def get_spectograms(file_list: list[str], rows=40, cols=100):
    arr = np.empty((len(file_list), rows, cols))

    for i in range(len(file_list)):
        file = file_list[i]
        if not file:
            continue

        if not os.path.isfile(file):
            print(f'{file} not found')
            sys.exit(1)

        y, fs = librosa.load(file, sr=16000)
        spectrogram = compute_spectrogram(y, fs)
        spectrogram = normalize(spectrogram)

        r, c = spectrogram.shape
        if r != rows or c != cols:
            spectrogram = np.pad(spectrogram, [(0, rows-r), (0, cols-c)],
                                 mode='constant', constant_values=0)

        arr[i] = normalize(spectrogram)
    return arr

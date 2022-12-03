#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import librosa
import soundfile as sf
import os

def open_list():
    lines = [
        'doing_the_dishes.wav',
        'dude_miaowing.wav',
        'exercise_bike.wav',
        'pink_noise.wav',
        'running_tap.wav',
        'white_noise.wav'
    ]
    return lines

def write_file(files, lists):
    f = open(lists, "w")
    f.writelines(files)
    f.close()

def clip_file(audio_file):
    y, sr = librosa.load(audio_file, sr=16000)
    samples = y.shape[0] // sr
    filename = os.path.splitext(os.path.basename(audio_file))[0]
    filename_list = []
    
    for i in range(samples * 5):
        offset = sr // 6 # Just a factor with overlaps
        si = i * offset
        ei = si + sr
        y_split = y[si:ei]
        output_filename = os.path.join('data','background_noise',f"{filename}_{i}.wav")
        sf.write(output_filename, y_split, 16000, 'PCM_16')
        filename_list.append(output_filename + '\n')
    
    return filename_list

if __name__ == "__main__":
    path = 'data/_background_noise_'
    os.makedirs('data/background_noise', exist_ok=True)
    files = open_list()
    filelist = []

    for i in files:
        filename = os.path.join(path, i).strip('\n')
        loc_list = clip_file(filename)

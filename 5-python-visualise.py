#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import os
from spectrogram import compute_spectrogram

def plot_signal_save(audio_file, output_file='signal.svg'):
    y, sr = librosa.load(audio_file)
    plt.figure();
    plt.plot(y);
    plt.title('Signal');
    plt.xlabel('Time (samples)');
    plt.ylabel('Amplitude');
    plt.tight_layout();
    plt.savefig(output_file);

def plot_spectrogram_save(audio_file, output_file='spectrogram.svg'):
    y, fs = librosa.load(audio_file);
    S_dB = compute_spectrogram(y, fs);
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=fs,
                         fmax=8192, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.tight_layout();
    plt.savefig(output_file);
    return S_dB

def plot_histogram_save(S_dB, output_file='distro.svg'):
    y, fs = librosa.load(audio_file);
    plt.figure();
    _ = plt.hist(S_dB, bins=100, density=True);
    plt.title("Histogram of the spectrum");
    plt.xlabel("Input Pixel Value")
    plt.ylabel("Probability Density")
    plt.tight_layout();
    plt.savefig(output_file);


if __name__ == "__main__":
    files = [
        {'cmd':'down', 'file':'data/training/down/0a7c2a8d_nohash_0'},
        {'cmd':'go', 'file':'data/training/go/0a9f9af7_nohash_0'},
        {'cmd':'left', 'file':'data/training/left/0a7c2a8d_nohash_0'},
        {'cmd':'no', 'file':'data/training/no/0a9f9af7_nohash_0'},
        {'cmd':'off', 'file':'data/training/off/0ab3b47d_nohash_1'},
        {'cmd':'on', 'file':'data/training/on/0a7c2a8d_nohash_0'},
        {'cmd':'right', 'file':'data/training/right/0a7c2a8d_nohash_0'},
        {'cmd':'stop', 'file':'data/training/stop/0ac15fe9_nohash_0'},
        {'cmd':'unknown', 'file':'data/training/unknown/0a7c2a8d_nohash_1'},
        {'cmd':'up', 'file':'data/training/up/0a7c2a8d_nohash_0'},
        {'cmd':'yes', 'file':'data/training/yes/0a7c2a8d_nohash_0'},
    ]
    S_dB = np.array([])

    # plot signals
    for i in files:
        audio_file = i["file"] + '.wav'
        
        base_file = os.path.splitext(os.path.basename(audio_file))[0] + '.svg'
        signal_file = os.path.join('img/audio', "signal_" + i["cmd"] + "_" + base_file)
        spectro_file = os.path.join('img/audio', "spectro_" + i["cmd"] + "_" + base_file)
        prob_file = os.path.join('img/audio', "prob_" + i["cmd"] + "_" + base_file)

        # Plot signal
        plot_signal_save(audio_file, signal_file)

        # Plot spectrogram
        S_dB_loc = plot_spectrogram_save(audio_file, spectro_file)
        S_dB = np.concatenate((S_dB, S_dB_loc.flatten()))

        # Plot distro
    plot_histogram_save(S_dB, "img/prob.svg")

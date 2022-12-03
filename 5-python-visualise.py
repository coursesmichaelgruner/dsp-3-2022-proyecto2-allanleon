#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import os
from spectrogram import compute_spectrogram
import glob

def plot_signal_save(audio_file, output_file='signal.svg'):
    y, sr = librosa.load(audio_file, sr=16000)
    plt.figure();
    plt.plot(y);
    plt.title('Signal');
    plt.xlabel('Time (samples)');
    plt.ylabel('Amplitude');
    plt.tight_layout();
    plt.savefig(output_file);

def plot_spectrogram_save(audio_file, output_file='spectrogram.svg'):
    y, fs = librosa.load(audio_file, sr=16000);
    S_dB = compute_spectrogram(y, fs);
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=fs,
                         fmax=fs//2, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.tight_layout();
    plt.savefig(output_file);
    return S_dB

def plot_histogram_save(S_dB, output_file='distro.svg'):
    y, fs = librosa.load(audio_file, sr=16000);
    plt.figure();
    _ = plt.hist(S_dB, bins=100, density=True);
    plt.title("Histogram of the spectrum");
    plt.xlabel("Input Pixel Value")
    plt.ylabel("Probability Density")
    plt.tight_layout();
    plt.savefig(output_file);


if __name__ == "__main__":
    cmds = [
        'down',
        'go',
        'left',
        'no',
        'off',
        'on',
        'right',
        'stop',
        'unknown',
        'up',
        'yes',
        'background_noise'
    ]
    S_dB = np.array([])

    # plot signals
    for cmd in cmds:
        audio_file = glob.glob(f'data/training/{cmd}/*.wav')
        if audio_file is None or len(audio_file)==0:
            print(f'no file for "{cmd}" found')
            continue
        audio_file=audio_file[0]

        base_file = os.path.splitext(os.path.basename(audio_file))[0] + '.svg'
        signal_file = os.path.join('img/audio', "signal_" + cmd + "_" + base_file)
        spectro_file = os.path.join('img/audio', "spectro_" + cmd + "_" + base_file)
        prob_file = os.path.join('img/audio', "prob_" + cmd + "_" + base_file)

        # Plot signal
        plot_signal_save(audio_file, signal_file)

        # Plot spectrogram
        S_dB_loc = plot_spectrogram_save(audio_file, spectro_file)
        S_dB = np.concatenate((S_dB, S_dB_loc.flatten()))

        # Plot distro
    plot_histogram_save(S_dB, "img/prob.svg")

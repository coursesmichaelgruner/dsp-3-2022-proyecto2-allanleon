#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

def compute_spectrogram(audiodata, fs=16000, coldur=0.025, coldist=0.01):
    winlen = int(fs * coldur)
    hoplen = int(fs * coldist) + 1
    nfft = 8192 # 8 kHz len
    win = 'hann' # Hanning window
    centred = True
    pow = 2 # Power mode
    n_mels = 40

    # Compute spectrum in power
    mel_spect = librosa.feature.melspectrogram(audiodata, sr=fs, n_fft=nfft,
        win_length=winlen, hop_length=hoplen,window=win, center=centred,
        power=pow, n_mels=n_mels)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    return mel_spect
    
    
def plot_spectrogram(S_dB, fs=16000, fmax=8000):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=fs,
                         fmax=fmax, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()

if __name__ == "__main__":
    y, fs = librosa.load('./data/training/down/0a7c2a8d_nohash_0.wav')
    S_dB = compute_spectrogram(y, fs)
    print(S_dB.shape)
    plot_spectrogram(S_dB, fs)

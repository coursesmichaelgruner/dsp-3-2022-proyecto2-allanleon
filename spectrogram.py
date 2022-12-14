import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

def compute_spectrogram(audiodata, fs=16000, coldur=0.025, coldist=0.01, ref=np.max, db=True):
    winlen = int(fs * coldur)
    hoplen = int(fs * coldist) + 1
    nfft = fs//2 # 8 kHz len
    win = 'hann' # Hanning window
    centred = True
    pow = 2 # Power mode
    n_mels = 40

    # Compute spectrum in power
    mel_spect_db = librosa.feature.melspectrogram(audiodata, sr=fs, n_fft=nfft,
        win_length=winlen, hop_length=hoplen,window=win, center=centred,
        power=pow, n_mels=n_mels)
    if db:
        mel_spect_db = librosa.power_to_db(mel_spect_db, ref=ref)
    return mel_spect_db
    
    
def plot_spectrogram(S_dB, fs=16000, fmax=8000):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=fs,
                         fmax=fmax, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()

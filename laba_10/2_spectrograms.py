import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.signal.windows import hann
import os

file_names = ['AAA.wav', 'III.wav', 'Barking.wav']
output_dir = 'result'
os.makedirs(output_dir, exist_ok=True)

nperseg = 1024
noverlap = 512
window = hann(nperseg, sym=False)

for filename in file_names:
    try:

        fs, audio = wavfile.read(filename)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32) / np.max(np.abs(audio))

        f, t, Sxx = spectrogram(audio, fs, window=window, nperseg=nperseg, noverlap=noverlap)
        f_log = f[1:]  
        Sxx_log = Sxx[1:, :]

        plt.figure(figsize=(10, 5))
        plt.pcolormesh(t, f_log, 10 * np.log10(Sxx_log + 1e-12),
                       shading='gouraud', cmap='inferno')
        plt.yscale('log')
        plt.ylim(20, fs/2)
        plt.xlabel('Время (с)')
        plt.ylabel('Частота (Гц) - log шкала')
        plt.colorbar(label='Энергия (дБ)')
        plt.title(f'Спектрограмма: {filename}')
        plt.tight_layout()

        save_name = os.path.join(output_dir, f'spectrogram_{filename[:-4]}.png')
        plt.savefig(save_name, dpi=150)
        plt.close()
        print(f' Построена спектрограмма для {filename}')

    except FileNotFoundError:
        print(f'Файл {filename} не найден. Пропускаем.')
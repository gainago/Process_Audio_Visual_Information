import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.signal.windows import hann
import os

file_names = ['AAA.wav', 'III.wav', 'Barking.wav']
output_dir = 'result'
os.makedirs(output_dir, exist_ok=True)

nperseg = 2048
noverlap = 1024
window = hann(nperseg, sym=False)
energy_threshold_db = -50  # порог для определения голоса

for filename in file_names:
    try:
        fs, audio = wavfile.read(filename)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32) / np.max(np.abs(audio))

        f, t, Sxx = spectrogram(audio, fs, window=window, nperseg=nperseg, noverlap=noverlap)
        f = f[1:]       # убираем 0 Гц
        Sxx = Sxx[1:, :]

        avg_spectrum_db = 10 * np.log10(np.mean(Sxx, axis=1) + 1e-12)

        above_threshold = avg_spectrum_db > energy_threshold_db
        if np.any(above_threshold):
            indices = np.where(above_threshold)[0]
            min_freq = f[indices[0]]
            max_freq = f[indices[-1]]
        else:
            min_freq = max_freq = 0

        result_file = os.path.join(output_dir, f'min_max_freq_{filename[:-4]}.txt')
        with open(result_file, 'w') as f_out:
            f_out.write(f'Минимальная частота голоса: {min_freq:.1f} Гц\n')
            f_out.write(f'Максимальная частота голоса: {max_freq:.1f} Гц\n')
        print(f'{filename}: min={min_freq:.1f} Гц, max={max_freq:.1f} Гц')

    except FileNotFoundError:
        print(f'Файл {filename} не найден. Пропускаем.')

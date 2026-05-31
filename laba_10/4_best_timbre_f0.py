import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
import os

file_names = ['AAA.wav', 'III.wav', 'Barking.wav']
output_dir = 'result'
os.makedirs(output_dir, exist_ok=True)

def best_timbre_f0(audio, fs, f0_range=(80, 400), step=5, n_harmonics=10):
    f, t, Sxx = spectrogram(audio, fs, nperseg=2048, noverlap=1024)
    f = f[1:]
    Sxx = Sxx[1:, :]
    avg_power = np.mean(Sxx, axis=1)

    best_f0 = 0
    best_score = -1

    for f0 in np.arange(f0_range[0], f0_range[1], step):

        idx_f0 = np.abs(f - f0).argmin()
        if idx_f0 >= len(avg_power):
            continue

        power_f0 = avg_power[idx_f0]

        sum_harmonics = 0
        for h in range(2, n_harmonics + 1):
            target_freq = f0 * h
            if target_freq > f[-1]:
                break
            idx_h = np.abs(f - target_freq).argmin()
            sum_harmonics += avg_power[idx_h]

        if power_f0 > 1e-12:
            score = sum_harmonics / power_f0
        else:
            score = 0

        if score > best_score:
            best_score = score
            best_f0 = f0

    return best_f0, best_score

for filename in file_names:
    try:
        fs, audio = wavfile.read(filename)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32) / np.max(np.abs(audio))

        f0, score = best_timbre_f0(audio, fs)

        result_file = os.path.join(output_dir, f'best_timbre_f0_{filename[:-4]}.txt')
        with open(result_file, 'w') as f_out:
            f_out.write(f'Наиболее тембрально окрашенный основной тон: {f0:.1f} Гц\n')
            f_out.write(f'Оценка обертональной насыщенности: {score:.4f}\n')
        print(f'{filename}: f0 = {f0:.1f} Гц (score={score:.4f})')

    except FileNotFoundError:
        print(f' Файл {filename} не найден. Пропускаем.')
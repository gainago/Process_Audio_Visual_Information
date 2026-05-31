import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.signal.windows import hann
import os

file_names = ['AAA.wav', 'III.wav', 'Barking.wav']
output_dir = 'result'
os.makedirs(output_dir, exist_ok=True)


dt = 0.1           
df = 10           
nperseg_dt = int(dt * 44100)  
noverlap_dt = 0

for filename in file_names:
    try:
        fs, audio = wavfile.read(filename)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32) / np.max(np.abs(audio))

        
        nperseg = int(dt * fs)
        if nperseg < 2:  
            nperseg = 1024
        window = hann(nperseg, sym=False)
        f, t, Sxx = spectrogram(audio, fs, window=window, nperseg=nperseg, noverlap=noverlap_dt)
        f = f[1:]      
        Sxx = Sxx[1:, :]

        band_edges = np.arange(20, fs/2, df)
        energy_bins = []

        for low in band_edges:
            high = low + df
            idx_f = np.where((f >= low) & (f < high))[0]
            if len(idx_f) == 0:
                continue
            total_energy = np.sum(Sxx[idx_f, :])
            energy_bins.append((low, high, total_energy))

        energy_bins.sort(key=lambda x: x[2], reverse=True)

        top_3 = energy_bins[:3]

        res_file = os.path.join(output_dir, f'formants_{filename[:-4]}.txt')
        with open(res_file, 'w') as f_out:
            f_out.write(f'Три самые сильные форманты (Δt={dt} с, Δf={df} Гц):\n')
            for i, (low, high, energy) in enumerate(top_3):
                f_out.write(f'{i+1}. {low:.1f}-{high:.1f} Гц (энергия: {energy:.2f})\n')
        print(f' {filename}: форманты записаны в {res_file}')

        plt.figure(figsize=(12, 6))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12),
                       shading='gouraud', cmap='inferno')
        plt.yscale('log')
        plt.ylim(20, 8000)  
        plt.xlabel('Время (с)')
        plt.ylabel('Частота (Гц) - log шкала')
        plt.colorbar(label='Энергия (дБ)')

        colors = ['cyan', 'lime', 'red']
        for i, (low, high, energy) in enumerate(top_3):
            mid_freq = (low + high) / 2
            plt.axhline(y=mid_freq, color=colors[i], linestyle='--', linewidth=2,
                        label=f'Форманта {i+1}: {low:.0f}-{high:.0f} Гц')
        plt.legend()
        plt.title(f'Три форманты для {filename}')
        plt.tight_layout()

        plot_file = os.path.join(output_dir, f'formants_plot_{filename[:-4]}.png')
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f'Построен график формант для {filename}')

    except FileNotFoundError:
        print(f' Файл {filename} не найден. Пропускаем.')

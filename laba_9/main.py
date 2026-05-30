import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram, stft, istft, wiener, savgol_filter, butter, filtfilt
from scipy.signal.windows import hann
import os


filename = 'Piano_Music.wav' 
fs, audio = wavfile.read(filename)

if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

audio = audio.astype(np.float32) / np.max(np.abs(audio))

print(f"Файл: {filename}")
print(f"Частота дискретизации: {fs} Гц")
print(f"Длительность: {len(audio)/fs:.3f} с")
print(f"Каналы: моно (после преобразования)")


nperseg = 1024
noverlap = 512
window = hann(nperseg, sym=False)

f, t, Sxx = spectrogram(audio, fs, window=window, nperseg=nperseg, noverlap=noverlap)

f_log = f[1:]
Sxx_log = Sxx[1:, :]

plt.figure(figsize=(12, 6))
plt.pcolormesh(t, f_log, 10 * np.log10(Sxx_log + 1e-12), shading='gouraud', cmap='inferno')
plt.yscale('log')
plt.ylim(20, fs/2)
plt.xlabel('Время (с)')
plt.ylabel('Частота (Гц) - лог. шкала')
plt.colorbar(label='Энергия (дБ)')
plt.title('Исходная спектрограмма')
plt.tight_layout()
plt.savefig('spectrogram_original.png', dpi=150)
plt.close()
print("Исходная спектрограмма сохранена как spectrogram_original.png")

noise_dur = 0.3
noise_len = int(noise_dur * fs)
threshold = 0.02

if np.max(np.abs(audio[:noise_len])) < threshold:
    noise_sample = audio[:noise_len]
    print("Участок шума взят из начала записи (тишина)")
elif np.max(np.abs(audio[-noise_len:])) < threshold:
    noise_sample = audio[-noise_len:]
    print("Участок шума взят из конца записи (тишина)")
else:
    noise_sample = audio[:noise_len]
    print("Тишина не найдена, шум оценён по первым 0.3 с")

_, _, S_noise = spectrogram(noise_sample, fs, window=window, nperseg=nperseg, noverlap=noverlap)
noise_spectrum = np.mean(S_noise, axis=1)

Sxx_clean = np.maximum(Sxx - noise_spectrum[:, np.newaxis], 0)

_, _, Zxx = stft(audio, fs, window=window, nperseg=nperseg, noverlap=noverlap,
                 boundary=None, padded=False)

magnitude_clean = np.sqrt(Sxx_clean)

Zxx_clean_complex = magnitude_clean * np.exp(1j * np.angle(Zxx))

_, audio_restored = istft(Zxx_clean_complex, fs, window=window, nperseg=nperseg, noverlap=noverlap)

if len(audio_restored) > len(audio):
    audio_restored = audio_restored[:len(audio)]
else:
    audio_restored = np.pad(audio_restored, (0, len(audio)-len(audio_restored)))

audio_restored = audio_restored / np.max(np.abs(audio_restored))

wavfile.write('restored_spectral_sub.wav', fs, (audio_restored * 32767).astype(np.int16))
print("Восстановленный сигнал (спектральное вычитание) сохранён в restored_spectral_sub.wav")

audio_savgol = savgol_filter(audio, window_length=51, polyorder=3)
audio_wiener = wiener(audio, mysize=5, noise=0.1)
def lowpass_filter(signal, cutoff=3000, fs=fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)
audio_lowpass = lowpass_filter(audio, cutoff=3000, fs=fs)

wavfile.write('filtered_savgol.wav', fs, (audio_savgol * 32767).astype(np.int16))
wavfile.write('filtered_wiener.wav', fs, (audio_wiener * 32767).astype(np.int16))
wavfile.write('filtered_lowpass.wav', fs, (audio_lowpass * 32767).astype(np.int16))
print("Файлы отфильтрованных версий: filtered_savgol.wav, filtered_wiener.wav, filtered_lowpass.wav")

_, _, Sxx_restored = spectrogram(audio_restored, fs, window=window, nperseg=nperseg, noverlap=noverlap)
plt.figure(figsize=(12, 6))
plt.pcolormesh(t, f_log, 10 * np.log10(Sxx_restored[1:, :] + 1e-12), shading='gouraud', cmap='inferno')
plt.yscale('log')
plt.ylim(20, fs/2)
plt.xlabel('Время (с)')
plt.ylabel('Частота (Гц) - лог. шкала')
plt.colorbar(label='Энергия (дБ)')
plt.title('Спектрограмма после спектрального вычитания')
plt.tight_layout()
plt.savefig('spectrogram_cleaned.png', dpi=150)
plt.close()
print("Спектрограмма после очистки сохранена как spectrogram_cleaned.png")

dt = 0.1
df = 45
nperseg_dt = int(dt * fs)
noverlap_dt = 0

window_dt = hann(nperseg_dt, sym=False)
f_dt, t_dt, Sxx_dt = spectrogram(audio, fs, window=window_dt, nperseg=nperseg_dt, noverlap=noverlap_dt)

band_edges = np.arange(20, fs/2, df)
energy_cells = []

for low in band_edges:
    high = low + df
    idx = np.where((f_dt >= low) & (f_dt < high))[0]
    if len(idx) == 0:
        continue
    for i_time, t_center in enumerate(t_dt):
        energy = np.sum(Sxx_dt[idx, i_time])
        energy_cells.append((t_center, low, high, energy))

energy_cells.sort(key=lambda x: x[3], reverse=True)

# Вывод топ-5
print("\nТоп-5 моментов с максимальной энергией (Δf={} Гц, Δt={} с):".format(df, dt))
for i, (t_c, f_low, f_high, en) in enumerate(energy_cells[:5]):
    print(f"   {i+1}. Время: {t_c:.3f} с, полоса {f_low:.0f}-{f_high:.0f} Гц, энергия: {en:.2f}")

plt.figure(figsize=(12, 6))
plt.pcolormesh(t, f_log, 10 * np.log10(Sxx_log + 1e-12), shading='gouraud', cmap='inferno')
plt.yscale('log')
plt.ylim(20, fs/2)
plt.xlabel('Время (с)')
plt.ylabel('Частота (Гц) - лог. шкала')
plt.colorbar(label='Энергия (дБ)')

colors = ['cyan', 'lime', 'yellow', 'white', 'magenta']
for i, (t_c, f_low, f_high, en) in enumerate(energy_cells[:5]):
    mid_freq = (f_low + f_high) / 2
    plt.plot(t_c, mid_freq, 'o', color=colors[i], markersize=10,
             markeredgecolor='black', label=f'Топ-{i+1}: {t_c:.3f}с, {f_low:.0f}-{f_high:.0f}Гц')
plt.legend()
plt.title('Топ-5 моментов с максимальной энергией')
plt.tight_layout()
plt.savefig('spectrogram_energy_moments.png', dpi=150)
plt.close()
print(" График с отмеченными моментами сохранён как spectrogram_energy_moments.png")

print("\n Лабораторная работа выполнена. Все результаты сохранены.")
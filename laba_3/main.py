import os
import numpy as np
from PIL import Image


def load_grayscale(path: str) -> np.ndarray:

    img = Image.open(path).convert('L')
    return np.array(img, dtype=np.uint8)

def save_image(arr: np.ndarray, path: str) -> None:

    Image.fromarray(arr.astype(np.uint8), mode='L').save(path)


def rank_filter_3x3_rank7_9(img: np.ndarray) -> np.ndarray:

    h, w = img.shape
    result = np.zeros((h, w), dtype=np.uint8)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            window = img[y-1:y+2, x-1:x+2].flatten()
            window_sorted = np.sort(window)
            result[y, x] = window_sorted[6]  # индекс 6 = 7-й элемент
    return result


if __name__ == '__main__':

    halftone_path = 'input/photo_halftone.png'
    monochrome_path = 'input/photo_monochrome.png' 

    print("\n--- Полутоновое изображение ---")
    gray = load_grayscale(halftone_path)

    gray_filtered = rank_filter_3x3_rank7_9(gray)
    save_image(gray_filtered, 'results/photo_filtered_halftone.png')
    print("Отфильтрованное полутоновое (ранг 7/9): results/photo_filtered_halftone.png")

    diff_gray = np.abs(gray.astype(int) - gray_filtered.astype(int)).astype(np.uint8)

    if np.max(diff_gray) < 30: 
        diff_gray = np.clip(diff_gray * 10, 0, 255).astype(np.uint8)
        print("Разностное полутоновое было тёмным, применено контрастирование (*10).")

    save_image(diff_gray, 'results/photo_diff_halftone.png')
    print("Разностное полутоновое (модуль разности): results/photo_diff_halftone.png")

    print("\n--- Монохромное изображение ---")
    mono = load_grayscale(monochrome_path)

    mono_filtered = rank_filter_3x3_rank7_9(mono)
    save_image(mono_filtered, 'results/photo_filtered_monochrome.png')
    print("Отфильтрованное монохромное (ранг 7/9): results/photo_filtered_monochrome.png")

    diff_mono = np.abs(mono.astype(int) - mono_filtered.astype(int)).astype(np.uint8) #это работает так же как xor
    save_image(diff_mono, 'results/photo_diff_monochrome.png')
    print("Разностное монохромное (хог): results/photo_diff_monochrome.png")

    print("\nЗадание выполнено полностью (оба пункта для полутона и монохрома).")
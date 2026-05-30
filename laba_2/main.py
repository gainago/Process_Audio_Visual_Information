import os
import numpy as np
from PIL import Image
from math import sqrt

def load_image(path: str) -> np.ndarray:

    img = Image.open(path).convert('RGB')
    return np.array(img, dtype=np.uint8)

def save_image(arr: np.ndarray, path: str) -> None:

    if arr.ndim == 2:
        img = Image.fromarray(arr.astype(np.uint8), mode='L')
    else:
        img = Image.fromarray(arr.astype(np.uint8), mode='RGB')
    img.save(path)


def to_grayscale(rgb: np.ndarray) -> np.ndarray:

    R = rgb[:, :, 0].astype(float)
    G = rgb[:, :, 1].astype(float)
    B = rgb[:, :, 2].astype(float)
    
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    return np.round(gray).astype(np.uint8)

def binarize_wan(gray: np.ndarray, window_size: int = 15, k: float = 0.2) -> np.ndarray:
    
    # Формула WAN T = mean_win + k * (std_win - global_std)

    h, w = gray.shape
    
    global_mean = np.mean(gray)
    global_std = sqrt(np.mean((gray - global_mean) ** 2))
    
    binary = np.zeros((h, w), dtype=np.uint8)
    
    half = window_size // 2
    for y in range(h):
        for x in range(w):
            y1 = max(0, y - half)
            y2 = min(h, y + half + 1)
            x1 = max(0, x - half)
            x2 = min(w, x + half + 1)
            
            window = gray[y1:y2, x1:x2]
            
            mean_win = np.mean(window)
            std_win = np.std(window)  
            
            threshold = mean_win + k * (std_win - global_std)
            
            if gray[y, x] > threshold:
                binary[y, x] = 255   # белый 
            else:
                binary[y, x] = 0     # чёрный 
    
    return binary


if __name__ == '__main__':

 input_names = ['contour_map', 'X-ray_image', 'cartoon_screenshot', 'photo', 'fingerprint', 'text_page']
ext = '.png'
prev = 'input/'

K_WAN = 0.2


for name in input_names:
    src_path = f"{prev}{name}{ext}"
    if not os.path.exists(src_path):
        print(f"Файл {src_path} не найден, пропускаем.")
        continue

    print(f"\nОбработка {src_path}...")
    original = load_image(src_path)

    gray = to_grayscale(original)
    save_image(gray, f"results/{name}_halftone.png")
    print(f"  Полутон сохранён: {name}_halftone.png")

    binary_3x3 = binarize_wan(gray, window_size=3, k=K_WAN)
    save_image(binary_3x3, f"results/{name}_monochrome_3x3.png")
    print(f"  Бинарное (WAN 3×3) сохранено: {name}_monochrome_3x3.png")

    binary_25x25 = binarize_wan(gray, window_size=25, k=K_WAN)
    save_image(binary_25x25, f"results/{name}_monochrome_25x25.png")
    print(f"  Бинарное (WAN 25×25) сохранено: {name}_monochrome_25x25.png")

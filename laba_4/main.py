import numpy as np
from PIL import Image
import os

def rgb_to_grayscale(rgb):

    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray

def convolve2d(image, kernel):
    H, W = image.shape
    kH, kW = kernel.shape
    pad_h = kH // 2
    pad_w = kW // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output = np.zeros((H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            region = padded[i:i+kH, j:j+kW]
            output[i, j] = np.sum(region * kernel)

    return output

def normalize_to_0_255(mat):
    min_val = np.min(mat)
    max_val = np.max(mat)
    if max_val == min_val:
        return np.zeros_like(mat, dtype=np.uint8)
    norm = (mat - min_val) * 255.0 / (max_val - min_val)
    return norm.astype(np.uint8)

def binarize(gradient, threshold):

    binary = np.where(gradient >= threshold, 255, 0).astype(np.uint8)
    return binary

def main():
    input_path = "./input/photo.png"
    result_dir = "./result"
    
    img = Image.open(input_path).convert('RGB')
    img_rgb = np.array(img)

    img_gray = rgb_to_grayscale(img_rgb)
    gray_save_path = os.path.join(result_dir, "photo_halftone.png")
    Image.fromarray(img_gray).save(gray_save_path)
    print(f"Сохранено полутоновое изображение: {gray_save_path}")

    # Ядра Прюитта 3×3
    kernel_gx = np.array([[1, 0, -1],
                          [1, 0, -1],
                          [1, 0, -1]], dtype=np.float32)

    kernel_gy = np.array([[1, 1, 1],
                          [ 0,  0,  0],
                          [-1,  -1,  -1]], dtype=np.float32)

    gx = convolve2d(img_gray, kernel_gx)
    gy = convolve2d(img_gray, kernel_gy)

    # G = |Gx| + |Gy|
    g = np.abs(gx) + np.abs(gy)

    # матрицу в [0, 255] 
    gx_norm = normalize_to_0_255(gx)
    gy_norm = normalize_to_0_255(gy)
    g_norm = normalize_to_0_255(g)

    gx_save_path = os.path.join(result_dir, "photo_gx.png")
    Image.fromarray(gx_norm).save(gx_save_path)
    print(f"Сохранена матрица Gx: {gx_save_path}")

    gy_save_path = os.path.join(result_dir, "photo_gy.png")
    Image.fromarray(gy_norm).save(gy_save_path)
    print(f"Сохранена матрица Gy: {gy_save_path}")

    g_save_path = os.path.join(result_dir, "photo_g.png")
    Image.fromarray(g_norm).save(g_save_path)
    print(f"Сохранена матрица G (|Gx|+|Gy|): {g_save_path}")
    for i in range(110, 10, -10):
        threshold = i
        binary_g = binarize(g, threshold)
        binary_save_path = os.path.join(result_dir, f"photo_binary{threshold}.png")
        Image.fromarray(binary_g).save(binary_save_path)
        print(f"Сохранена бинаризованная G (порог {threshold}): {binary_save_path}")

if __name__ == "__main__":
    main()
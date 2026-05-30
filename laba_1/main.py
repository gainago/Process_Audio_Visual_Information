import os
import numpy as np
from PIL import Image
from math import acos, sqrt, pi

def load_image(path: str) -> np.ndarray:

    img = Image.open(path).convert('RGB')
    return np.array(img, dtype=np.uint8)

def save_image(arr: np.ndarray, path: str) -> None:

    img = Image.fromarray(arr.astype(np.uint8))
    img.save(path)


def extract_channels(rgb: np.ndarray) -> tuple:

    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]
    return R, G, B

def rgb_to_hsi(rgb: np.ndarray) -> tuple:

    R = rgb[:, :, 0].astype(float) / 255.0
    G = rgb[:, :, 1].astype(float) / 255.0
    B = rgb[:, :, 2].astype(float) / 255.0

    I = (R + G + B) / 3.0

    min_RGB = np.minimum(np.minimum(R, G), B)
    sum_RGB = R + G + B
    S = np.zeros_like(I)
    mask = sum_RGB > 0
    S[mask] = 1.0 - 3.0 * min_RGB[mask] / sum_RGB[mask]

    H = np.zeros_like(I)
    mask_s = S > 0
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B) * (G - B))

    den[den == 0] = 1e-10
    cos_theta = np.clip(num / den, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    H[mask_s] = theta[mask_s] * 180.0 / pi

    mask_bg = B > G
    H[mask_s & mask_bg] = 360.0 - H[mask_s & mask_bg]

    H_norm = H / 360.0 * 255.0
    S_norm = S * 255.0
    I_norm = I * 255.0

    return H_norm.astype(np.uint8), S_norm.astype(np.uint8), I_norm.astype(np.uint8)

def hsi_to_rgb(H: np.ndarray, S: np.ndarray, I: np.ndarray) -> np.ndarray:

    H = H.astype(float) / 255.0 * 360.0   # градусы
    S = S.astype(float) / 255.0
    I = I.astype(float) / 255.0

    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)

    mask0 = (0 <= H) & (H < 120)
    H0 = H[mask0]
    B[mask0] = I[mask0] * (1 - S[mask0])
    R[mask0] = I[mask0] * (1 + S[mask0] * np.cos(np.radians(H0)) / np.cos(np.radians(60 - H0)))
    G[mask0] = 3 * I[mask0] - (R[mask0] + B[mask0])

    mask1 = (120 <= H) & (H < 240)
    H1 = H[mask1] - 120
    R[mask1] = I[mask1] * (1 - S[mask1])
    G[mask1] = I[mask1] * (1 + S[mask1] * np.cos(np.radians(H1)) / np.cos(np.radians(60 - H1)))
    B[mask1] = 3 * I[mask1] - (R[mask1] + G[mask1])

    mask2 = (240 <= H) & (H < 360)
    H2 = H[mask2] - 240
    G[mask2] = I[mask2] * (1 - S[mask2])
    B[mask2] = I[mask2] * (1 + S[mask2] * np.cos(np.radians(H2)) / np.cos(np.radians(60 - H2)))
    R[mask2] = 3 * I[mask2] - (G[mask2] + B[mask2])

    rgb = np.stack([R, G, B], axis=2)
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return rgb

def invert_intensity(rgb: np.ndarray) -> np.ndarray:

    H, S, I = rgb_to_hsi(rgb)
    I_inv = 255 - I                     # инверсия яркости
    return hsi_to_rgb(H, S, I_inv)

def stretch_image(img: np.ndarray, M: int) -> np.ndarray:

    h, w, ch = img.shape
    new_h = int(round(h * M))
    new_w = int(round(w * M))
    new_img = np.zeros((new_h, new_w, ch), dtype=img.dtype)

    for y in range(new_h):
        for x in range(new_w):
            # ближайший пиксель
            src_y = min(int(round(y / M)), h - 1)
            src_x = min(int(round(x / M)), w - 1)
            new_img[y, x] = img[src_y, src_x]
    return new_img

def decimate_image(img: np.ndarray, N: int) -> np.ndarray:

    h, w, ch = img.shape
    new_h = int(round(h / N))
    new_w = int(round(w / N))
    new_img = np.zeros((new_h, new_w, ch), dtype=img.dtype)

    for y in range(new_h):
        for x in range(new_w):
            src_y = min(y * N, h - 1)
            src_x = min(x * N, w - 1)
            new_img[y, x] = img[src_y, src_x]
    return new_img

def two_pass_resampling(img: np.ndarray, M: int, N: int) -> np.ndarray:

    tmp = stretch_image(img, M)
    return decimate_image(tmp, N)

def one_pass_resampling(img: np.ndarray, K: float) -> np.ndarray:

    h, w, ch = img.shape
    new_h = int(round(h * K))
    new_w = int(round(w * K))
    new_img = np.zeros((new_h, new_w, ch), dtype=img.dtype)

    for y in range(new_h):
        for x in range(new_w):
            src_y = min(int(round(y / K)), h - 1)
            src_x = min(int(round(x / K)), w - 1)
            new_img[y, x] = img[src_y, src_x]
    return new_img


if __name__ == '__main__':
   
    src_path = "input/input_2.png"

    original = load_image(src_path)
    print(f"Изображение загружено. Размер: {original.shape}")

    os.makedirs("results", exist_ok=True)

    R, G, B = extract_channels(original)

    R_color = np.zeros_like(original)
    R_color[:, :, 0] = R
    save_image(R_color, "results/R_channel.png")
    G_color = np.zeros_like(original)
    G_color[:, :, 1] = G
    save_image(G_color, "results/G_channel.png")

    B_color = np.zeros_like(original)
    B_color[:, :, 2] = B
    save_image(B_color, "results/B_channel.png")

    print("Цветные каналы R, G, B сохранены.")

    H, S, I = rgb_to_hsi(original)
    save_image(I, "results/I_component.png")
    print("Яркостная компонента (I) сохранена.")

    inverted_img = invert_intensity(original)
    save_image(inverted_img, "results/inverted_brightness.png")
    print("Изображение с инвертированной яркостью сохранено.")

    print("\n Передискретизация")

    M = 2
    stretched = stretch_image(original, M)
    save_image(stretched, f"results/stretched_M{M}.png")
    print(f"Растянутое изображение (M={M}) сохранено.")

    N = 3
    decimated = decimate_image(original, N)
    save_image(decimated, f"results/decimated_N{N}.png")
    print(f"Сжатое изображение (N={N}) сохранено.")

    K = M / N
    print(f"Двухпроходная передискретизация с K = {K} (M={M}, N={N})")
    two_pass = two_pass_resampling(original, M, N)
    save_image(two_pass, f"results/two_pass_M{M}_N{N}.png")
    print("Результат двухпроходной передискретизации сохранён.")

    K_float = 0.75
    one_pass = one_pass_resampling(original, K_float)
    save_image(one_pass, f"results/one_pass_K{K_float}.png")
    print("Результат однопроходной передискретизации сохранён.")

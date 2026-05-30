import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


os.makedirs('output', exist_ok=True)


def rgb_to_hsl(image):
    hsl = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    return hsl[:, :, 1]

def quantize_image(img, levels=32):
    img = np.array(img, dtype=np.float32)
    img = (img / 255.0) * (levels - 1)
    return np.round(img).astype(np.uint8)

def compute_glrlm(img, directions=[0, 45, 90, 135], levels=32):
    h, w = img.shape
    max_run = max(h, w)
    glrlm_sum = np.zeros((levels, max_run), dtype=np.uint32)

    for angle in directions:
        angle_rad = np.deg2rad(angle)
        dx = int(round(np.cos(angle_rad)))
        dy = int(round(np.sin(angle_rad)))
        if dx == 0 and dy == 0:
            continue

        glrlm = np.zeros((levels, max_run), dtype=np.uint32)

        for y in range(h):
            for x in range(w):
                if (0 <= y - dy < h and 0 <= x - dx < w and
                    img[y, x] == img[y - dy, x - dx]):
                    continue

                run_length = 1
                ny, nx = y + dy, x + dx
                while 0 <= ny < h and 0 <= nx < w and img[ny, nx] == img[y, x]:
                    run_length += 1
                    ny += dy
                    nx += dx

                gray_level = int(img[y, x])
                if gray_level < levels:
                    if run_length >= max_run:
                        run_length = max_run - 1
                    glrlm[gray_level, run_length - 1] += 1

        glrlm_sum += glrlm

    return glrlm_sum

def glrlm_features(matrix):
    total_runs = np.sum(matrix)
    if total_runs == 0:
        return 0.0, 0.0

    row_sum = np.sum(matrix, axis=1)
    col_sum = np.sum(matrix, axis=0)

    glnu = np.sum(row_sum ** 2) / total_runs
    rlnu = np.sum(col_sum ** 2) / total_runs
    return glnu, rlnu

def linear_contrast(img):
    img_float = img.astype(np.float32)
    min_val = np.min(img_float)
    max_val = np.max(img_float)
    if max_val - min_val == 0:
        return img
    stretched = (img_float - min_val) * 255.0 / (max_val - min_val)
    return np.clip(stretched, 0, 255).astype(np.uint8)

def visualize_glrlm_adaptive(matrix, title, ax=None):
    if ax is None:
        ax = plt.gca()
    
    matrix_log = np.log1p(matrix.astype(np.float64))
    
   
    if np.sum(matrix) == 0:
        ax.imshow(np.zeros_like(matrix_log), cmap='gray')
        ax.set_title(f"{title} (все нули)")
        return
    
  
    non_zero = matrix_log[matrix_log > 0]
    if len(non_zero) > 0:
        vmax = np.percentile(non_zero, 95)  
    else:
        vmax = 1.0
    
  
    im = ax.imshow(matrix_log, cmap='gray', aspect='auto', vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel('Длина серии')
    ax.set_ylabel('Уровень серого')
    
   
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


image_files = [
    'input/cartoon_screenshot.png',
    'input/photo.png',
    'input/photo_2.png',
    'input/screenshot.png',
    'input/text_page.png'
]

levels = 32

for image_file in image_files:
    print(f"\n{'='*50}")
    print(f"Обработка: {image_file}")
    print('='*50)

   
    img_rgb = cv2.imread(image_file)
    if img_rgb is None:
        print(f"ОШИБКА: Не удалось загрузить {image_file}. Пропускаем.")
        continue
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

   
    L_original = rgb_to_hsl(img_rgb)

   
    L_contrasted = linear_contrast(L_original)

    
    L_quant_orig = quantize_image(L_original, levels)
    L_quant_contr = quantize_image(L_contrasted, levels)

   
    glrlm_orig = compute_glrlm(L_quant_orig, levels=levels)
    glnu_orig, rlnu_orig = glrlm_features(glrlm_orig)

    glrlm_contr = compute_glrlm(L_quant_contr, levels=levels)
    glnu_contr, rlnu_contr = glrlm_features(glrlm_contr)

    print(f"Исходное:    GLNU = {glnu_orig:.2f},  RLNU = {rlnu_orig:.2f}")
    print(f"Контрастированное: GLNU = {glnu_contr:.2f},  RLNU = {rlnu_contr:.2f}")

    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Результаты для: {os.path.basename(image_file)}', fontsize=16)

    
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Исходное (RGB)')
    axes[0, 0].axis('off')

    
    axes[0, 1].imshow(L_original, cmap='gray')
    axes[0, 1].set_title('Исходное полутоновое (L)')
    axes[0, 1].axis('off')

    
    axes[0, 2].imshow(L_contrasted, cmap='gray')
    axes[0, 2].set_title('Контрастированное')
    axes[0, 2].axis('off')

    
    axes[0, 3].axis('off')
    text_info = (f"Исходное:\n"
                 f"GLNU = {glnu_orig:.2f}\n"
                 f"RLNU = {rlnu_orig:.2f}\n\n"
                 f"Контрастированное:\n"
                 f"GLNU = {glnu_contr:.2f}\n"
                 f"RLNU = {rlnu_contr:.2f}")
    axes[0, 3].text(0.1, 0.5, text_info, fontsize=12, verticalalignment='center')

   
    axes[1, 0].hist(L_original.ravel(), bins=256, range=(0, 255), color='gray')
    axes[1, 0].set_title('Гистограмма исходного')

    
    axes[1, 1].hist(L_contrasted.ravel(), bins=256, range=(0, 255), color='gray')
    axes[1, 1].set_title('Гистограмма контрастированного')

    
    visualize_glrlm_adaptive(glrlm_orig, 'GLRLM (исходное)', ax=axes[1, 2])

   
    visualize_glrlm_adaptive(glrlm_contr, 'GLRLM (контрастированное)', ax=axes[1, 3])

    plt.tight_layout()
    
    
    base_name = os.path.splitext(os.path.basename(image_file))[0]
    output_filename = f'output/result_{base_name}.png'
    plt.savefig(output_filename, dpi=150)
    print(f"График сохранён: {output_filename}")
    plt.close()

print("\nВсе изображения обработаны! Результаты сохранены в папке 'output'.")
import os
import numpy as np
from PIL import Image

input_path = "input/screenshot.png"
output_path = "result/monochrome_transparent.bmp"

os.makedirs("result", exist_ok=True)

img = Image.open(input_path).convert('RGBA')
data = np.array(img)

r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
gray = 0.299 * r + 0.587 * g + 0.114 * b

threshold = 200
mask = gray < threshold   # буквы (тёмные) → True

# Создаём 1-битное изображение без использования режима '1' в fromarray
mono_data = np.ones((data.shape[0], data.shape[1]), dtype=np.uint8)
mono_data[mask] = 0   # буквы – чёрные (0)

# Преобразуем в 8-битное серое, затем в монохром
gray_img = Image.fromarray((mono_data * 255).astype(np.uint8), 'L')
new_img = gray_img.convert('1')   # конвертация в монохром

# Обрезка
bbox = new_img.getbbox()
if bbox is not None:
    new_img = new_img.crop(bbox)

new_img.save(output_path)
print(f"Готово! Сохранено: {output_path}")
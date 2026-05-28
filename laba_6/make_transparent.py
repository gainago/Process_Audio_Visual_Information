import os
import numpy as np
from PIL import Image

# Пути
input_path = "input/screenshot.png"
output_path = "result/monochrome_transparent.png"

os.makedirs("result", exist_ok=True)

img = Image.open(input_path).convert('RGBA')
data = np.array(img)

r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]

gray = 0.299 * r + 0.587 * g + 0.114 * b

threshold = 200
mask = gray < threshold  

new_data = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.uint8)

new_data[mask, 0] = 0
new_data[mask, 1] = 0
new_data[mask, 2] = 0
new_data[mask, 3] = 255  

new_data[~mask, 0] = 255
new_data[~mask, 1] = 255
new_data[~mask, 2] = 255
new_data[~mask, 3] = 0    

new_img = Image.fromarray(new_data, 'RGBA')
new_img.save(output_path)

print(f"Готово! Сохранено: {output_path}")
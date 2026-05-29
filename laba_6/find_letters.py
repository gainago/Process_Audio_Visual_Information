import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

img_path = "result/monochrome_transparent_72.bmp"
img = Image.open(img_path).convert('RGBA') 
arr = np.array(img)
mask = (arr[:, :, 0] < 128).astype(int)     

h_profile = mask.sum(axis=1)

def find_blocks(profile, threshold=0, min_len=1):
    blocks = []
    start = None
    for i, val in enumerate(profile):
        if val > threshold and start is None:
            start = i
        elif val <= threshold and start is not None:
            if i - start >= min_len:
                blocks.append((start, i-1))
            start = None
    if start is not None and len(profile) - start >= min_len:
        blocks.append((start, len(profile)-1))
    return blocks

rows = find_blocks(h_profile, threshold=1, min_len=5)

all_chars = []
for y1, y2 in rows:
    row_mask = mask[y1:y2+1, :]
    v_profile = row_mask.sum(axis=0)
    cols = find_blocks(v_profile, threshold=1, min_len=3)
    for x1, x2 in cols:
        all_chars.append((x1, y1, x2, y2))

all_chars.sort(key=lambda b: (b[1], b[0]))

print(f"Найдено строк: {len(rows)}")
print(f"Найдено символов: {len(all_chars)}")

draw = ImageDraw.Draw(img)
for (x1, y1, x2, y2) in all_chars:
    # Толщина линии = 2 пикселя, цвет = зелёный (0, 255, 0)
    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 255), width=2)

output_path = "result/segmented_on_original_72.png"
img.save(output_path)
print(f"Сохранено: {output_path}")

with open("segmentation_coords_72.csv", "w") as f:
    f.write("x1;y1;x2;y2\n")
    for x1, y1, x2, y2 in all_chars:
        f.write(f"{x1};{y1};{x2};{y2}\n")
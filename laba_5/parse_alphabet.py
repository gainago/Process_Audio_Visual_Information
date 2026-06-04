import os
import csv
import numpy as np
from PIL import Image

input_image_path = "Alphabet_72.bmp"
segmentation_csv = "Alphabet_segmentation_coords_72.csv"
output_dir = "osmanya_chars_72"
ground_truth = "𐒀𐒁𐒂𐒃𐒄𐒅𐒆𐒇𐒈𐒉𐒊𐒋𐒌𐒍𐒎𐒏𐒐𐒑𐒒𐒓𐒔𐒕𐒖𐒗𐒘𐒙𐒚𐒛𐒜𐒝"

os.makedirs(output_dir, exist_ok=True)

img = Image.open(input_image_path).convert('1')
img_array = np.array(img, dtype=np.uint8)

with open(segmentation_csv, 'r') as f:
    reader = csv.reader(f, delimiter=';')
    next(reader)
    bboxes = [(int(row[0]), int(row[1]), int(row[2]), int(row[3])) for row in reader]

for idx, (x1, y1, x2, y2) in enumerate(bboxes, start=1):
    subimg = img_array[y1:y2+1, x1:x2+1]
    char = ground_truth[idx - 1]                 # символ из ground_truth
    code = ord(char)                             # десятичный код
    filename = f"U{code:04X}_{char}.bmp"         # например: U10480_𐒀.bmp
    filepath = os.path.join(output_dir, filename)
    Image.fromarray((subimg * 255).astype(np.uint8)).convert('L').save(filepath)

print(f"Сохранено {len(bboxes)} файлов в '{output_dir}'")
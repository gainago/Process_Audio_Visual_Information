import os
from PIL import Image, ImageDraw, ImageFont

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
FONT_SIZE = 52

FONT_PATH = "LiberationSerif-Regular.ttf"

output_folder = "english_chars"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

try:
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    print("Шрифт загружен успешно!")
except OSError:
    print(f"Ошибка: Шрифт {FONT_PATH} не найден.")
    print("Убедитесь, что вы установили fonts-liberation через sudo apt install fonts-liberation.")
    exit()

for char in LETTERS:
    img = Image.new('RGBA', (200, 200), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    draw.text((0, 0), char, font=font, fill=(0, 0, 0, 255))
    
    bbox = img.getbbox()
    
    if bbox:
        cropped_img = img.crop(bbox)
        file_path = os.path.join(output_folder, f"{char}.png")
        cropped_img.save(file_path)
        print(f"Сохранена: {char}")
    else:
        print(f"Ошибка с буквой: {char}")

print("Готово!")
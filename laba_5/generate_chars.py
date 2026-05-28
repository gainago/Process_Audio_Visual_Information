import os
from PIL import Image, ImageDraw, ImageFont

OSMANYA_LETTERS = "𐒀𐒁𐒂𐒃𐒄𐒅𐒆𐒇𐒈𐒉𐒊𐒋𐒌𐒍𐒎𐒏𐒐𐒑𐒒𐒓𐒔𐒕𐒖𐒗𐒘𐒙𐒚𐒛𐒜𐒝"

FONT_SIZE = 52
FONT_PATH = "NotoSansOsmanya-Regular.ttf"

output_folder = "osmanya_chars"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

try:
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    print("Шрифт Noto Sans Osmanya загружен успешно!")
except OSError:
    print(f"Ошибка: Шрифт {FONT_PATH} не найден.")
    print("Установите его командой: sudo apt install fonts-noto-extra")
    exit()

for char in OSMANYA_LETTERS:
    img = Image.new('RGBA', (200, 200), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    draw.text((0, 0), char, font=font, fill=(0, 0, 0, 255))
    
    bbox = img.getbbox()
    
    if bbox:
        cropped_img = img.crop(bbox)
        codepoint = f"U{ord(char):05X}"
        file_path = os.path.join(output_folder, f"{codepoint}_{char}.png")
        cropped_img.save(file_path)
        print(f"Сохранён символ: {char} → {file_path}")
    else:
        print(f"Ошибка: буква {char} не имеет видимых пикселей.")

print("Генерация завершена!")
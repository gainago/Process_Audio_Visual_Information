import os
from PIL import Image, ImageDraw, ImageFont

OSMANYA_LETTERS = "𐒀𐒁𐒂𐒃𐒄𐒅𐒆𐒇𐒈𐒉𐒊𐒋𐒌𐒍𐒎𐒏𐒐𐒑𐒒𐒓𐒔𐒕𐒖𐒗𐒘𐒙𐒚𐒛𐒜𐒝"

FONT_SIZE = 52
FONT_PATH = "NotoSansOsmanya-Regular.ttf"

output_folder = "osmanya_chars_72"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

try:
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    print("Шрифт Noto Sans Osmanya загружен успешно!")
except OSError:
    exit()

for char in OSMANYA_LETTERS:
    img = Image.new('L', (300, 300), 255)
    draw = ImageDraw.Draw(img)
    
    draw.text((0, 0), char, font=font, fill=0)
    
    img_inv = img.point(lambda x: 255 - x)
    
    bbox = img_inv.getbbox()
    
    if bbox:
        cropped_inv = img_inv.crop(bbox)
        
        cropped = cropped_inv.point(lambda x: 255 - x)

        binary = cropped.point(lambda x: 0 if x < 128 else 255, '1')

        codepoint = f"U{ord(char):05X}"
        file_path = os.path.join(output_folder, f"{codepoint}_{char}.bmp")
        binary.save(file_path)
        print(f"Сохранён символ: {char} → {file_path} (размер: {binary.size})")
    else:
        print(f"Ошибка: буква {char} не имеет видимых пикселей.")

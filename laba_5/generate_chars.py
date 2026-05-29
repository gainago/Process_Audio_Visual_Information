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
    # 1. Создаём изображение в оттенках серого (L) с белым фоном (255)
    img = Image.new('L', (300, 300), 255)
    draw = ImageDraw.Draw(img)
    
    # 2. Рисуем букву чёрным цветом (0)
    draw.text((0, 0), char, font=font, fill=0)
    
    # 3. Инвертируем: буква становится белой (255), фон — чёрным (0)
    img_inv = img.point(lambda x: 255 - x)
    
    # 4. Находим bounding box по светлым пикселям (теперь это буква)
    bbox = img_inv.getbbox()
    
    if bbox:
        # 5. Обрезаем инвертированное изображение по bbox
        cropped_inv = img_inv.crop(bbox)
        
        # 6. Инвертируем обратно: буква снова чёрная, фон белый
        cropped = cropped_inv.point(lambda x: 255 - x)
        
        # 7. Применяем порог (бинаризация): всё, что темнее 128, становится чёрным (0),
        #    остальное — белым (255)
        #    Это создаёт чёткие контуры без сглаживания
        binary = cropped.point(lambda x: 0 if x < 128 else 255, '1')
        
        # 8. Сохраняем как монохромный BMP
        codepoint = f"U{ord(char):05X}"
        file_path = os.path.join(output_folder, f"{codepoint}_{char}.bmp")
        binary.save(file_path)
        print(f"Сохранён символ: {char} → {file_path} (размер: {binary.size})")
    else:
        print(f"Ошибка: буква {char} не имеет видимых пикселей.")

print("Генерация завершена!")
import os
from PIL import Image, ImageDraw, ImageFont

# Фраза для генерации
TEXT = "𐒀 𐒁 𐒂 𐒃 𐒄 𐒅 𐒆 𐒇 𐒈 𐒉 𐒊 𐒋 𐒌 𐒍 𐒎 𐒏 𐒐 𐒑 𐒒 𐒓 𐒔 𐒕 𐒖 𐒗 𐒘 𐒙 𐒚 𐒛 𐒜 𐒝"
FONT_SIZE = 72
FONT_PATH = "NotoSansOsmanya-Regular.ttf"

output_file = "Alphabet_72.bmp"

try:
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    print("Шрифт Noto Sans Osmanya загружен успешно!")
except OSError:
    print("Ошибка: шрифт не найден.")
    exit()

# Создаём большое изображение (достаточно широкое и высокое)
img = Image.new('L', (2000, 500), 255)
draw = ImageDraw.Draw(img)

# Рисуем текст с отступом слева и сверху (например, 10 пикселей)
draw.text((10, 10), TEXT, font=font, fill=0)

# Инвертируем для поиска bbox (чёрный текст на белом фоне → белый текст на чёрном)
img_inv = img.point(lambda x: 255 - x)

# Находим bounding box вокруг всех непрозрачных пикселей (текста)
bbox = img_inv.getbbox()

if bbox:
    # Обрезаем по bbox
    cropped_inv = img_inv.crop(bbox)
    # Возвращаем нормальные цвета (чёрный текст на белом фоне)
    cropped = cropped_inv.point(lambda x: 255 - x)
    # Преобразуем в монохромный BMP (1 бит на пиксель)
    binary = cropped.point(lambda x: 0 if x < 128 else 255, '1')
    binary.save(output_file)
    print(f"Фраза сохранена: {output_file} (размер: {binary.size})")
else:
    print("Ошибка: текст не содержит видимых пикселей.")
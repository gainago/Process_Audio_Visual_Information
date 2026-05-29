import os
import numpy as np
from PIL import Image
from scipy.spatial.distance import euclidean

import os
import matplotlib.pyplot as plt

def save_symbol_profile(img, output_path, title=""):
    """
    Сохраняет горизонтальный и вертикальный профили буквы в виде картинки.
    img: PIL Image (режим 'L' или 'RGBA')
    output_path: путь для сохранения (например, 'profile.png')
    """
    arr = np.array(img.convert('L'))
    mask = (arr < 128).astype(int)
    
    h_profile = mask.sum(axis=1)
    v_profile = mask.sum(axis=0)
    
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(h_profile)), h_profile, color='black')
    plt.title(f"{title} - гор. профиль")
    plt.xlabel("Строка")
    plt.ylabel("Сумма")
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(v_profile)), v_profile, color='black')
    plt.title(f"{title} - верт. профиль")
    plt.xlabel("Столбец")
    plt.ylabel("Сумма")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

def save_features_csv(features, output_path, symbol=""):
    """
    Сохраняет вектор признаков в CSV-файл.
    features: numpy array из 6 элементов: масса, cx_norm, cy_norm, Ixx_norm, Iyy_norm, Ixy_norm
    output_path: путь для сохранения
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("feature,value\n")
        f.write(f"mass,{features[0]}\n")
        f.write(f"cx_norm,{features[1]}\n")
        f.write(f"cy_norm,{features[2]}\n")
        f.write(f"Ixx_norm,{features[3]}\n")
        f.write(f"Iyy_norm,{features[4]}\n")
        f.write(f"Ixy_norm,{features[5]}\n")

def save_cropped_symbol(image_path, bbox, output_path):
    """
    Вырезает символ по bbox и сохраняет на прозрачном фоне.
    image_path: путь к исходному монохромному изображению (RGBA)
    bbox: (x1, y1, x2, y2)
    output_path: куда сохранить PNG
    """
    img = Image.open(image_path).convert('RGBA')
    x1, y1, x2, y2 = bbox
    cropped = img.crop((x1, y1, x2, y2))
    cropped.save(output_path, format='PNG')

def export_reference_profiles(ref_dir, out_root="result/signs"):
    """
    Для каждого эталона создаёт папку result/signs/символ/ с profile.png и features.csv.
    """
    os.makedirs(out_root, exist_ok=True)
    for fname in os.listdir(ref_dir):
        if not (fname.endswith(".png") and '_' in fname):
            continue
        char = fname.split('_')[1].replace(".png", "")
        path = os.path.join(ref_dir, fname)
        img = Image.open(path)
        features = extract_features_from_image(img)
        
        # Папка для символа
        char_dir = os.path.join(out_root, char)
        os.makedirs(char_dir, exist_ok=True)
        
        # Профиль
        profile_path = os.path.join(char_dir, "profile.png")
        save_symbol_profile(img, profile_path, f"Эталон {char}")
        
        # Признаки
        features_path = os.path.join(char_dir, "features.csv")
        save_features_csv(features, features_path, char)
    
    print(f"Эталонные профили и признаки сохранены в {out_root}")

def export_segmented_symbols(image_path, coords, out_root="result/parse_input"):
    """
    Для каждого сегментированного символа создаёт папку letter_N/
    с profile.png, features.csv, letter.png (вырезанная буква).
    Возвращает список векторов признаков (в том же порядке, что и coords).
    """
    os.makedirs(out_root, exist_ok=True)
    img = Image.open(image_path).convert('RGBA')
    arr = np.array(img)
    mask = (arr[:, :, 0] < 128).astype(int)
    
    features_list = []
    for idx, (x1, y1, x2, y2) in enumerate(coords, 1):
        letter_dir = os.path.join(out_root, f"letter_{idx}")
        os.makedirs(letter_dir, exist_ok=True)
        
        # Вырезаем символ
        char_mask = mask[y1:y2+1, x1:x2+1]
        char_img = Image.fromarray((char_mask * 255).astype(np.uint8))
        features = extract_features_from_image(char_img)
        features_list.append(features)
        
        # Сохраняем вырезанную букву (на прозрачном фоне)
        letter_png_path = os.path.join(letter_dir, "letter.png")
        save_cropped_symbol(image_path, (x1, y1, x2, y2), letter_png_path)
        
        # Профиль
        profile_path = os.path.join(letter_dir, "profile.png")
        save_symbol_profile(char_img, profile_path, f"Символ {idx}")
        
        # Признаки
        features_path = os.path.join(letter_dir, "features.csv")
        save_features_csv(features, features_path, f"letter_{idx}")
    
    print(f"Сегментированные символы сохранены в {out_root}")
    return features_list

# ================== НАСТРОЙКИ ==================
REF_DIR = "../laba_5/osmanya_chars"               # папка с эталонными буквами
INPUT_IMAGE = "../laba_6/result/monochrome_transparent.png"
COORDS_CSV = "../laba_6/segmentation_coords.csv"
OUTPUT_HYPOTHESES = "hypotheses.txt"
GROUND_TRUTH = "𐒖𐒒𐒏𐒚𐒃𐒗𐒋𐒐𐒖𐒔𐒖𐒕𐒒𐒝𐒐𐒝𐒈𐒔𐒖𐒖𐒆𐒖𐒕𐒂𐒖𐒔𐒖𐒕"   # в алфавите нет пробелов 

# Размер, к которому приводим все символы (высота 40 пикселей, ширина пропорциональна)
TARGET_HEIGHT = 40

# ================== ФУНКЦИИ ==================

def extract_features_from_image(img):
    """
    Извлекает признаки из бинарного изображения (PIL Image, режим 'L' или '1').
    Возвращает вектор признаков: [масса, нормированный cx, нормированный cy, Ixx, Iyy, Ixy]
    Предварительно изображение ресайзится к TARGET_HEIGHT с сохранением пропорций,
    затем паддится до квадрата максимального размера (чтобы избежать искажений).
    """
    # Приводим к чёрно-белому: чёрные пиксели = 1, белые = 0
    arr = np.array(img.convert('L'))
    mask = (arr < 128).astype(int)

    # Находим bounding box ненулевых пикселей
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    cropped = mask[y_min:y_max+1, x_min:x_max+1]

    # Ресайз с сохранением пропорций до TARGET_HEIGHT
    h, w = cropped.shape
    scale = TARGET_HEIGHT / h
    new_w = int(w * scale)
    if new_w == 0:
        new_w = 1
    pil_crop = Image.fromarray((cropped * 255).astype(np.uint8))
    pil_resized = pil_crop.resize((new_w, TARGET_HEIGHT), Image.Resampling.NEAREST)
    resized_arr = np.array(pil_resized) > 128  # бинарный

    # Паддинг до квадрата (максимальная сторона) – для единообразия моментов
    max_side = max(TARGET_HEIGHT, new_w)
    pad_h = max_side - TARGET_HEIGHT
    pad_w = max_side - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padded = np.pad(resized_arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

    # Масса
    mass = np.sum(padded)

    if mass == 0:
        # Защита от пустого символа
        return np.zeros(6)

    # Центр тяжести (пиксельные координаты внутри паддед-изображения)
    y_coords, x_coords = np.indices(padded.shape)
    cx = np.sum(x_coords * padded) / mass
    cy = np.sum(y_coords * padded) / mass

    # Нормировка координат относительно размеров изображения (0..1)
    h_pad, w_pad = padded.shape
    cx_norm = cx / w_pad
    cy_norm = cy / h_pad

    # Центрированные моменты инерции (вторые центральные моменты)
    # Ixx = Σ (y - cy)^2, Iyy = Σ (x - cx)^2, Ixy = Σ (x - cx)(y - cy)
    y_centered = y_coords - cy
    x_centered = x_coords - cx
    Ixx = np.sum(padded * y_centered**2) / mass
    Iyy = np.sum(padded * x_centered**2) / mass
    Ixy = np.sum(padded * x_centered * y_centered) / mass

    # Дополнительная нормализация моментов – делим на (размер^2) для масштабной инвариантности
    scale_norm = (h_pad * w_pad)  # или max_side^2
    Ixx_norm = Ixx / scale_norm
    Iyy_norm = Iyy / scale_norm
    Ixy_norm = Ixy / scale_norm

    return np.array([mass, cx_norm, cy_norm, Ixx_norm, Iyy_norm, Ixy_norm])


def load_reference_features(ref_dir):
    """
    Загружает все эталонные изображения из папки, извлекает признаки.
    Возвращает словарь {символ: вектор_признаков}
    """
    ref_features = {}
    for fname in os.listdir(ref_dir):
        if fname.endswith(".png") and '_' in fname:
            # Извлекаем символ из имени: например, "U1048A_ሀ.png" -> "ሀ"
            char = fname.split('_')[1].replace(".png", "")
            path = os.path.join(ref_dir, fname)
            img = Image.open(path)
            features = extract_features_from_image(img)
            ref_features[char] = features
    return ref_features


def read_segmentation_coords(csv_path):
    """
    Читает CSV с координатами, возвращает список (x1, y1, x2, y2).
    Формат: x1;y1;x2;y2 (без заголовка).
    """
    coords = []
    with open(csv_path, 'r') as f:
        lines = f.readlines()[1:]  # пропускаем заголовок
        for line in lines:
            parts = line.strip().split(';')
            if len(parts) == 4:
                x1, y1, x2, y2 = map(int, parts)
                coords.append((x1, y1, x2, y2))
    return coords


def extract_segmented_features(image_path, coords):
    """
    Извлекает признаки для каждого сегментированного символа.
    Возвращает список векторов признаков (в том же порядке, что и coords).
    """
    img = Image.open(image_path).convert('RGBA')
    arr = np.array(img)
    mask = (arr[:, :, 0] < 128).astype(int)  # чёрные пиксели = 1

    features_list = []
    for (x1, y1, x2, y2) in coords:
        # Вырезаем символ
        char_mask = mask[y1:y2+1, x1:x2+1]
        # Преобразуем в бинарное изображение для функции извлечения
        char_img = Image.fromarray((char_mask * 255).astype(np.uint8))
        features = extract_features_from_image(char_img)
        features_list.append(features)
    return features_list


def compute_similarity(feat1, feat2):
    """Евклидово расстояние, мера близости = 1 / (1 + distance)"""
    dist = euclidean(feat1, feat2)
    sim = 1.0 / (1.0 + dist)
    return sim


def recognize_characters(seg_features, ref_features):
    """
    Для каждого сегментированного символа вычисляет меры близости со всеми эталонами.
    Возвращает список списков гипотез: [(символ, similarity), ...] (отсортирован по убыванию similarity).
    """
    hypotheses_list = []
    for seg_feat in seg_features:
        hyp = []
        for char, ref_feat in ref_features.items():
            sim = compute_similarity(seg_feat, ref_feat)
            hyp.append((char, sim))
        hyp.sort(key=lambda x: x[1], reverse=True)
        hypotheses_list.append(hyp)
    return hypotheses_list


def save_hypotheses(hyp_list, out_path):
    """Сохраняет гипотезы в текстовый файл."""
    with open(out_path, 'w', encoding='utf-8') as f:
        for idx, hyp in enumerate(hyp_list):
            # Формат: "1: [('a', 0.99), ('o', 0.87), ...]"
            hyp_str = ", ".join([f"('{char}', {sim:.4f})" for char, sim in hyp])
            f.write(f"{idx+1}: [{hyp_str}]\n")
    print(f"Гипотезы сохранены в {out_path}")


def get_recognized_string(hyp_list):
    """Возвращает строку из лучших гипотез."""
    return ''.join([hyp[0][0] for hyp in hyp_list])


def evaluate_accuracy(ground, recognized):
    """Вычисляет количество ошибок и процент верных."""
    if len(ground) != len(recognized):
        print("Предупреждение: длина строк разная, сравнение по минимальной длине.")
        min_len = min(len(ground), len(recognized))
        ground = ground[:min_len]
        recognized = recognized[:min_len]
    errors = sum(1 for g, r in zip(ground, recognized) if g != r)
    total = len(ground)
    accuracy = (total - errors) / total * 100 if total > 0 else 0
    return errors, accuracy


# ================== ОСНОВНАЯ ЧАСТЬ ==================

def main():
    # 1. Загружаем эталонные признаки и экспортируем их профили/признаки
    print("Загрузка и экспорт эталонных признаков...")
    export_reference_profiles(REF_DIR, "result/signs")
    ref_features = load_reference_features(REF_DIR)
    print(f"Загружено {len(ref_features)} эталонов.")

    # 2. Читаем координаты сегментированных символов
    if not os.path.exists(COORDS_CSV):
        print("Файл координат не найден. Запустите find_letters.py сначала.")
        return
    coords = read_segmentation_coords(COORDS_CSV)
    print(f"Найдено {len(coords)} сегментированных символов.")

    # 3. Экспорт сегментированных символов (признаки, профили, вырезанные буквы)
    print("Экспорт сегментированных символов...")
    seg_features = export_segmented_symbols(INPUT_IMAGE, coords, "result/parse_input")

    # 4. Распознавание – получаем гипотезы
    print("Распознавание...")
    hypotheses = recognize_characters(seg_features, ref_features)

    # 5. Сохраняем гипотезы
    save_hypotheses(hypotheses, OUTPUT_HYPOTHESES)

    # 6. Получаем лучшую распознанную строку
    recognized = get_recognized_string(hypotheses)
    print(f"Распознанная строка: {recognized}")

    # 7. Сравнение с ground truth
    ground = GROUND_TRUTH
    if ground:
        errors, accuracy = evaluate_accuracy(ground, recognized)
        print(f"Количество ошибок: {errors}")
        print(f"Процент верно распознанных: {accuracy:.2f}%")
    else:
        print("Не задана строка истины (GROUND_TRUTH), оценка не производится.")
if __name__ == '__main__':
    main()
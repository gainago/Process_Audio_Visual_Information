import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.distance import euclidean

# ================== НАСТРОЙКИ ==================
REF_DIR = "../laba_5/osmanya_chars"               # папка с эталонными буквами
INPUT_IMAGE = "../laba_6/result/monochrome_transparent.png"
COORDS_CSV = "../laba_6/segmentation_coords.csv"
OUTPUT_HYPOTHESES = "hypotheses.txt"
GROUND_TRUTH = "𐒖𐒒𐒏𐒚𐒃𐒗𐒋𐒐𐒖𐒔𐒖𐒕𐒒𐒝𐒐𐒝𐒈𐒔𐒖𐒖𐒆𐒖𐒕𐒂𐒖𐒔𐒖𐒕"

# Размер, к которому приводим все символы
TARGET_HEIGHT = 40

# ================== ФУНКЦИИ ==================

def extract_features_from_image(img):
    """Извлекает нормализованные признаки из бинарного изображения."""
    arr = np.array(img.convert('L'))
    mask = (arr < 128).astype(int)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    cropped = mask[y_min:y_max+1, x_min:x_max+1]

    h, w = cropped.shape
    scale = TARGET_HEIGHT / h
    new_w = int(w * scale)
    if new_w == 0:
        new_w = 1
    pil_crop = Image.fromarray((cropped * 255).astype(np.uint8))
    pil_resized = pil_crop.resize((new_w, TARGET_HEIGHT), Image.Resampling.NEAREST)
    resized_arr = np.array(pil_resized) > 128

    max_side = max(TARGET_HEIGHT, new_w)
    pad_h = max_side - TARGET_HEIGHT
    pad_w = max_side - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padded = np.pad(resized_arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

    mass = np.sum(padded)
    if mass == 0:
        return np.zeros(6)

    y_coords, x_coords = np.indices(padded.shape)
    cx = np.sum(x_coords * padded) / mass
    cy = np.sum(y_coords * padded) / mass

    h_pad, w_pad = padded.shape
    cx_norm = cx / w_pad
    cy_norm = cy / h_pad

    y_centered = y_coords - cy
    x_centered = x_coords - cx
    Ixx = np.sum(padded * y_centered**2) / mass
    Iyy = np.sum(padded * x_centered**2) / mass
    Ixy = np.sum(padded * x_centered * y_centered) / mass

    scale_norm = (h_pad * w_pad)
    Ixx_norm = Ixx / scale_norm
    Iyy_norm = Iyy / scale_norm
    Ixy_norm = Ixy / scale_norm

    return np.array([mass, cx_norm, cy_norm, Ixx_norm, Iyy_norm, Ixy_norm])

def save_symbol_profile(img, output_path, title=""):
    """Сохраняет профили (гор/верт) в виде картинки."""
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

def save_features_csv(features, output_path):
    """Сохраняет вектор признаков в CSV."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("feature,value\n")
        f.write(f"mass,{features[0]}\n")
        f.write(f"cx_norm,{features[1]}\n")
        f.write(f"cy_norm,{features[2]}\n")
        f.write(f"Ixx_norm,{features[3]}\n")
        f.write(f"Iyy_norm,{features[4]}\n")
        f.write(f"Ixy_norm,{features[5]}\n")

def save_cropped_symbol(image_path, bbox, output_path):
    """Вырезает символ по bbox и сохраняет на прозрачном фоне."""
    img = Image.open(image_path).convert('RGBA')
    x1, y1, x2, y2 = bbox
    cropped = img.crop((x1, y1, x2, y2))
    cropped.save(output_path, format='PNG')

def export_reference_profiles(ref_dir, out_root="result/signs"):
    """Сохраняет эталоны: symbol.png, profile.png, features.csv"""
    os.makedirs(out_root, exist_ok=True)
    for fname in os.listdir(ref_dir):
        if not (fname.endswith(".png") and '_' in fname):
            continue
        char = fname.split('_')[1].replace(".png", "")
        path = os.path.join(ref_dir, fname)
        img = Image.open(path)
        features = extract_features_from_image(img)
        
        char_dir = os.path.join(out_root, char)
        os.makedirs(char_dir, exist_ok=True)
        
        # Оригинальная буква
        img.save(os.path.join(char_dir, "symbol.png"))
        # Профиль
        save_symbol_profile(img, os.path.join(char_dir, "profile.png"), f"Эталон {char}")
        # Признаки
        save_features_csv(features, os.path.join(char_dir, "features.csv"))
    
    print(f"Эталонные файлы сохранены в {out_root}")

def load_reference_features(ref_dir):
    """Загружает признаки эталонов."""
    ref_features = {}
    for fname in os.listdir(ref_dir):
        if not (fname.endswith(".png") and '_' in fname):
            continue
        char = fname.split('_')[1].replace(".png", "")
        path = os.path.join(ref_dir, fname)
        img = Image.open(path)
        ref_features[char] = extract_features_from_image(img)
    return ref_features

def read_segmentation_coords(csv_path):
    """Читает координаты сегментированных символов."""
    coords = []
    with open(csv_path, 'r') as f:
        lines = f.readlines()[1:]  # пропускаем заголовок
        for line in lines:
            parts = line.strip().split(';')
            if len(parts) == 4:
                coords.append(tuple(map(int, parts)))
    return coords

def export_segmented_symbols(image_path, coords, out_root="result/parse_input"):
    """Сохраняет сегментированные символы: letter.png, profile.png, features.csv"""
    os.makedirs(out_root, exist_ok=True)
    img = Image.open(image_path).convert('RGBA')
    arr = np.array(img)
    mask = (arr[:, :, 0] < 128).astype(int)
    
    features_list = []
    for idx, (x1, y1, x2, y2) in enumerate(coords, 1):
        letter_dir = os.path.join(out_root, f"letter_{idx}")
        os.makedirs(letter_dir, exist_ok=True)
        
        char_mask = mask[y1:y2+1, x1:x2+1]
        char_img = Image.fromarray((char_mask * 255).astype(np.uint8))
        features = extract_features_from_image(char_img)
        features_list.append(features)
        
        # Вырезанная буква (прозрачный фон)
        save_cropped_symbol(image_path, (x1, y1, x2, y2), os.path.join(letter_dir, "letter.png"))
        # Профиль
        save_symbol_profile(char_img, os.path.join(letter_dir, "profile.png"), f"Символ {idx}")
        # Признаки
        save_features_csv(features, os.path.join(letter_dir, "features.csv"))
    
    print(f"Сегментированные символы сохранены в {out_root}")
    return features_list

def compute_similarity(feat1, feat2):
    dist = euclidean(feat1, feat2)
    return 1.0 / (1.0 + dist)

def recognize_characters(seg_features, ref_features):
    hypotheses_list = []
    for seg_feat in seg_features:
        hyp = [(char, compute_similarity(seg_feat, ref_feat)) for char, ref_feat in ref_features.items()]
        hyp.sort(key=lambda x: x[1], reverse=True)
        hypotheses_list.append(hyp)
    return hypotheses_list

def save_hypotheses(hyp_list, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for idx, hyp in enumerate(hyp_list):
            hyp_str = ", ".join([f"('{char}', {sim:.4f})" for char, sim in hyp])
            f.write(f"{idx+1}: [{hyp_str}]\n")
    print(f"Гипотезы сохранены в {out_path}")

def get_recognized_string(hyp_list):
    return ''.join([hyp[0][0] for hyp in hyp_list])

def evaluate_accuracy(ground, recognized):
    if len(ground) != len(recognized):
        print("Предупреждение: длина строк разная, сравнение по минимальной длине.")
        min_len = min(len(ground), len(recognized))
        ground = ground[:min_len]
        recognized = recognized[:min_len]
    errors = sum(1 for g, r in zip(ground, recognized) if g != r)
    total = len(ground)
    accuracy = (total - errors) / total * 100.1  if total > 0 else 0
    return errors, accuracy

# ================== ОСНОВНАЯ ЧАСТЬ ==================

if __name__ == "__main__":
    print("Экспорт эталонных букв...")
    export_reference_profiles(REF_DIR, "result/signs")
    
    ref_features = load_reference_features(REF_DIR)
    print(f"Загружено {len(ref_features)} эталонов.")
    
    if not os.path.exists(COORDS_CSV):
        print("Файл координат не найден. Запустите find_letters.py сначала.")
        exit(1)
    
    coords = read_segmentation_coords(COORDS_CSV)
    print(f"Найдено {len(coords)} сегментированных символов.")
    
    print("Экспорт сегментированных символов...")
    seg_features = export_segmented_symbols(INPUT_IMAGE, coords, "result/parse_input")
    
    print("Распознавание...")
    hypotheses = recognize_characters(seg_features, ref_features)
    save_hypotheses(hypotheses, OUTPUT_HYPOTHESES)
    
    recognized = get_recognized_string(hypotheses)
    print(f"Распознанная строка: {recognized}")
    
    if GROUND_TRUTH:
        errors, accuracy = evaluate_accuracy(GROUND_TRUTH, recognized)
        print(f"Количество ошибок: {errors}")
        print(f"Процент верно распознанных: {accuracy:.2f}%")
    else:
        print("Не задана строка истины, оценка не производится.")
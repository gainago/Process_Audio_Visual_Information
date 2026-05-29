import os
import csv
import numpy as np
from PIL import Image

# ========== Параметры ==========
input_image_path = "../laba_6/result/monochrome_transparent.bmp"
segmentation_csv = "../laba_6/segmentation_coords.csv"
alphabet_features_csv = "../laba_5/osmanya_features.csv"
alphabet_images_dir = "../laba_5/osmanya_chars"   # нужен только для NCC
output_dir = "result"
ground_truth = "𐒖𐒒𐒏𐒚𐒃𐒗𐒋𐒐𐒖𐒔𐒖𐒕𐒒𐒝𐒐𐒝𐒈𐒔𐒖𐒖𐒆𐒖𐒕𐒂𐒖𐒔𐒖𐒕"

# ========== ВЫБОР МЕТРИКИ ==========
similarity_metric = 'ncc'  # 'euclidean_features' или 'ncc'
# ====================================

os.makedirs(output_dir, exist_ok=True)

# ========== 1. Загрузка изображения и сегментации ==========
img = Image.open(input_image_path).convert('1')  # 1-битный монохром
img_array = np.array(img, dtype=np.uint8)  # 0 = чёрный (буква), 1 = белый (фон)

with open(segmentation_csv, 'r') as f:
    reader = csv.reader(f, delimiter=';')
    next(reader)
    bboxes = [(int(row[0]), int(row[1]), int(row[2]), int(row[3])) for row in reader]

print(f"Найдено {len(bboxes)} символов для распознавания.")

# ========== 2. Загрузка эталонов (в зависимости от метрики) ==========
alphabet_features = []   # для признаковой метрики
alphabet_images = {}     # для NCC

if similarity_metric == 'euclidean_features':
    # Загружаем признаки из CSV
    with open(alphabet_features_csv, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            # Извлекаем чистый символ (отбрасываем префикс U104xx_)
            full_letter = row['Letter']
            letter = full_letter.split('_')[-1]  # например, '𐒗'
            
            mass = int(row['Q1']) + int(row['Q2']) + int(row['Q3']) + int(row['Q4'])
            area = int(row['W']) * int(row['H'])
            mass_norm = mass / area if area > 0 else 0
            cx_rel = float(row['cx_rel'])
            cy_rel = float(row['cy_rel'])
            Ix_norm = float(row['Ix_norm'])
            Iy_norm = float(row['Iy_norm'])
            alphabet_features.append({
                'letter': letter,
                'features': np.array([mass_norm, cx_rel, cy_rel, Ix_norm, Iy_norm])
            })
else:
    # Загружаем бинарные изображения эталонов (для NCC)
    for fname in os.listdir(alphabet_images_dir):
        if fname.endswith('.bmp'):
            # Извлекаем чистый символ из имени файла (отбрасываем префикс)
            full_name = fname.replace('.bmp', '')
            letter = full_name.split('_')[-1]  # например, '𐒗'
            
            img_letter = Image.open(os.path.join(alphabet_images_dir, fname)).convert('1')
            alphabet_images[letter] = np.array(img_letter, dtype=np.uint8)  # 0 = чёрный, 1 = белый

# ========== 3. Вспомогательные функции для NCC ==========
def resize_to_fixed_size(binary_img, size=(32, 32)):
    """Приводит бинарное изображение к фиксированному размеру (32x32)"""
    pil_img = Image.fromarray((binary_img * 255).astype(np.uint8))
    resized = pil_img.resize(size, Image.Resampling.NEAREST)
    return np.array(resized) // 255

def ncc_similarity(img1, img2):
    """Нормированная кросс-корреляция для бинарных изображений (0/1)"""
    a = 2 * img1.astype(np.float32) - 1
    b = 2 * img2.astype(np.float32) - 1
    mu_a = np.mean(a)
    mu_b = np.mean(b)
    a_centered = a - mu_a
    b_centered = b - mu_b
    numerator = np.sum(a_centered * b_centered)
    denominator = np.sqrt(np.sum(a_centered**2) * np.sum(b_centered**2))
    if denominator == 0:
        return 0
    return numerator / denominator

# ========== 4. Обработка каждого символа ==========
all_hypotheses = []
best_letters = []

for idx, (x1, y1, x2, y2) in enumerate(bboxes, start=1):
    subimg = img_array[y1:y2+1, x1:x2+1]  # 0 = чёрный, 1 = белый
    
    # Сохраняем вырезку
    letter_dir = os.path.join(output_dir, f"letter_{idx}")
    os.makedirs(letter_dir, exist_ok=True)
    Image.fromarray((subimg * 255).astype(np.uint8)).convert('L').save(
        os.path.join(letter_dir, "section.bmp")
    )
    
    if similarity_metric == 'euclidean_features':
        # ---- Евклидово расстояние по признакам ----
        height, width = subimg.shape
        binary = 1 - subimg  # буква = 1, фон = 0
        y_coords, x_coords = np.where(binary == 1)
        mass = np.sum(binary)
        area = width * height
        mass_norm = mass / area if area > 0 else 0
        if mass == 0:
            cx_rel = 0.5
            cy_rel = 0.5
            Ix_norm = 0
            Iy_norm = 0
        else:
            cx = np.mean(x_coords)
            cy = np.mean(y_coords)
            cx_rel = cx / width
            cy_rel = cy / height
            Ix = np.sum((y_coords - cy) ** 2)
            Iy = np.sum((x_coords - cx) ** 2)
            Ix_norm = Ix / area
            Iy_norm = Iy / area
        sample_features = np.array([mass_norm, cx_rel, cy_rel, Ix_norm, Iy_norm])
        
        similarities = []
        for entry in alphabet_features:
            dist = np.linalg.norm(sample_features - entry['features'])
            similarity = 1 / (1 + dist)  # нулевое расстояние → 1
            similarities.append((entry['letter'], similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        
    else:  # similarity_metric == 'ncc'
        # ---- Нормированная кросс-корреляция ----
        subimg_fixed = resize_to_fixed_size(subimg)
        similarities = []
        for letter, ref_img in alphabet_images.items():
            ref_fixed = resize_to_fixed_size(ref_img)
            sim = ncc_similarity(subimg_fixed, ref_fixed)
            similarities.append((letter, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
    
    all_hypotheses.append(similarities)
    best_letters.append(similarities[0][0])

# ========== 5. Сохранение гипотез ==========
with open(os.path.join(output_dir, "hypotheses_ncc.txt"), 'w', encoding='utf-8') as f:
    for i, hyp in enumerate(all_hypotheses, start=1):
        hyp_str = ", ".join([f"(\"{letter}\", {sim:.3f})" for letter, sim in hyp])
        f.write(f"{i}: [{hyp_str}]\n")

print(f"Гипотезы сохранены в result/hypotheses_ncc.txt (метрика: {similarity_metric})")

# ========== 6. Лучшие гипотезы и сравнение с истиной ==========
best_string = ''.join(best_letters)
print("\n===== Лучшие гипотезы (первый столбец) =====")
print(best_string)

if ground_truth:
    truth_chars = [ch for ch in ground_truth if not ch.isspace()]
    total = len(truth_chars)
    correct = 0
    errors = []
    for i, (best, truth) in enumerate(zip(best_letters, truth_chars)):
        if best == truth:
            correct += 1
        else:
            errors.append((i+1, best, truth))
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n===== Сравнение с истиной =====")
    print(f"Всего символов: {total}")
    print(f"Верно: {correct}")
    print(f"Ошибок: {total - correct}")
    print(f"Точность: {accuracy:.1f}%")
    if errors:
        print("Ошибки (позиция, распознано, истина):")
        for pos, b, t in errors:
            print(f"  {pos}: '{b}' вместо '{t}'")
else:
    print("\nИстинная строка не введена, сравнение не выполнено.")

print(f"\nРезультаты сохранены в папке {output_dir}")
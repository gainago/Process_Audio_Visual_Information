import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

INPUT_FOLDER = "osmanya_chars_72"
OUTPUT_CSV = "osmanya_features_72.csv"
OUTPUT_PROFILES_FOLDER = "osmanya_profiles_72"

if not os.path.exists(OUTPUT_PROFILES_FOLDER):
    os.makedirs(OUTPUT_PROFILES_FOLDER)

def calculate_features(image_path):

    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    mask = (img_array < 128).astype(int)
    
    H, W = mask.shape 
    
    mid_x = W // 2
    mid_y = H // 2
    if mid_x == 0 or mid_y == 0:
        return None, None 

    q1 = mask[0:mid_y, 0:mid_x].sum()
    q2 = mask[0:mid_y, mid_x:W].sum()
    q3 = mask[mid_y:H, 0:mid_x].sum()
    q4 = mask[mid_y:H, mid_x:W].sum()
    
    quarter_area = mid_x * mid_y
    q1_rel = q1 / quarter_area
    q2_rel = q2 / quarter_area
    q3_rel = q3 / quarter_area
    q4_rel = q4 / quarter_area
    
    total_weight = mask.sum()
    if total_weight == 0:
        return None, None
    
    y_indices, x_indices = np.indices((H, W))
    
    cx = (mask * x_indices).sum() / total_weight
    cy = (mask * y_indices).sum() / total_weight
    
    cx_rel = cx / W
    cy_rel = cy / H
    
    Ix = (mask * (y_indices - cy)**2).sum()
    Iy = (mask * (x_indices - cx)**2).sum()
    
    area = W * H
    Ix_norm = Ix / area
    Iy_norm = Iy / area
    
    profile_x = mask.sum(axis=0) 

    profile_y = mask.sum(axis=1) 
    
    return {
        "W": W, "H": H,
        "Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4,
        "Q1_rel": q1_rel, "Q2_rel": q2_rel, "Q3_rel": q3_rel, "Q4_rel": q4_rel,
        "cx": cx, "cy": cy,
        "cx_rel": cx_rel, "cy_rel": cy_rel,
        "Ix": Ix, "Iy": Iy,
        "Ix_norm": Ix_norm, "Iy_norm": Iy_norm,
        "profile_x": profile_x, 
        "profile_y": profile_y
    }, (profile_x, profile_y)

def plot_and_save_profile(profile, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(10, 4))
    bars = plt.bar(range(len(profile)), profile, width=1.0, color='black')
    
    plt.xticks(np.arange(0, len(profile), step=max(1, len(profile)//10)))
    plt.yticks(np.arange(0, max(profile)+1, step=max(1, max(profile)//5)))
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

if __name__ == "__main__":
    data_rows = []
    
    header = ("Letter;W;H;Q1;Q2;Q3;Q4;Q1_rel;Q2_rel;Q3_rel;Q4_rel;"
              "cx;cy;cx_rel;cy_rel;Ix;Iy;Ix_norm;Iy_norm")
    
    files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith('.bmp')])
    
    print(f"Найдено {len(files)} символов, анализ...")
    
    for filename in files:
        letter = os.path.splitext(filename)[0]
        path = os.path.join(INPUT_FOLDER, filename)
        
        result, profiles = calculate_features(path)
        
        if result is None:
            print(f"Пропущен {letter} (размер слишком мал для деления на четверти)")
            continue
            

        row = [letter, result['W'], result['H'], 
               result['Q1'], result['Q2'], result['Q3'], result['Q4'],
               result['Q1_rel'], result['Q2_rel'], result['Q3_rel'], result['Q4_rel'],
               result['cx'], result['cy'], result['cx_rel'], result['cy_rel'],
               result['Ix'], result['Iy'], result['Ix_norm'], result['Iy_norm']]
        data_rows.append(row)
        
        profile_x, profile_y = profiles
        
        # Извлекаем только код Unicode (например, U10480) для заголовка, чтобы не было предупреждений
        letter_display = letter.split('_')[0]   # или просто letter.replace('_', ' ')
        
        plot_and_save_profile(
            profile_x, 
            f"Вертикальный профиль символа '{letter_display}'", 
            "Номер столбца", "Сумма чёрных пикселей",
            os.path.join(OUTPUT_PROFILES_FOLDER, f"{letter}_profile_x.png")
        )
        
        plot_and_save_profile(
            profile_y, 
            f"Горизонтальный профиль символа '{letter_display}'", 
            "Номер строки", "Сумма чёрных пикселей",
            os.path.join(OUTPUT_PROFILES_FOLDER, f"{letter}_profile_y.png")
        )
        
        print(f"Обработан символ: {letter}")
    
    with open(OUTPUT_CSV, 'w', encoding='utf-8') as f:
        f.write(header + '\n')
        for row in data_rows:
            row_str = ';'.join(str(v) for v in row)
            f.write(row_str + '\n')
    
    print(f"\nГотово! Данные сохранены в '{OUTPUT_CSV}'.")
    print(f"Профили сохранены в папке '{OUTPUT_PROFILES_FOLDER}'.")
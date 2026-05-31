import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator 
from PIL import Image

def plot_char_profiles(img_path, save_path_prefix):
    img = Image.open(img_path).convert('L')
    arr = np.array(img)
    mask = (arr < 128).astype(int)
    
    h_profile = mask.sum(axis=1)  
    v_profile = mask.sum(axis=0)  
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(h_profile)), h_profile, color='black')
    plt.title("Горизонтальный профиль (по строкам)")
    plt.xlabel("Строка")
    plt.ylabel("Сумма")
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(v_profile)), v_profile, color='black')
    plt.title("Вертикальный профиль (по столбцам)")
    plt.xlabel("Столбец")
    plt.ylabel("Сумма")
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}.png")
    plt.close()

source_dir = "../laba_5/osmanya_chars"
output_dir = "alphabet_profiles"
os.makedirs(output_dir, exist_ok=True)

for fname in sorted(os.listdir(source_dir)):
    if fname.endswith(".bmp"):
        name = fname.replace(".bmp", "")
        path = os.path.join(source_dir, fname)
        plot_char_profiles(path, os.path.join(output_dir, f"profile_{name}"))

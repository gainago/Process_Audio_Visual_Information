# Лабораторная работа №4. Выделение контуров на изображении

## Вариант 8: Оператор Прюитт 3×3

#### Преобразование полноцветного изображения в полутоновое

Использовалось взвешенное усреднение каналов RGB:

$$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$

#### Ядра оператора Прюитт 3×3

$$G_x = \begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1
\end{bmatrix} \quad
G_y = \begin{bmatrix}
1 & 1 & 1 \\
 0 &  0 &  0 \\
-1 & -1 & -1
\end{bmatrix}$$

#### Модуль градиента

$$G = |G_x| + |G_y|$$

---

### 1. Исходное и полутоновое изображения

| Исходное (RGB) | Полутоновое |
|:--------------:|:------------:|
| ![original](input/photo.png) | ![grayscale](result/photo_halftone.png) |

---

### 2. Градиентные матрицы \(G_x\), \(G_y\) и модуль градиента \(G\)

| \(G_x\) | \(G_y\) | \(G\) |
|:-------:|:-------:|:-----:|
| ![gx](result/photo_gx.png) | ![gy](result/photo_gy.png) | ![g](result/photo_g.png) |


### 3. Сравнение порогов бинаризации

Для демонстрации влияния порога на результат можно получить бинарные изображения при различных значениях \(T\).  

| Порог | Результат бинаризации |
|:-----:|:----------------------:|
| **T=20** | ![binary](result/photo_binary20.png) |
| **T=30** | ![binary](result/photo_binary30.png)|
| **T=40** | ![binary](result/photo_binary40.png) |
| **T=50** | ![binary](result/photo_binary50.png) |
| **T=60** | ![binary](result/photo_binary60.png) |
| **T=70** | ![binary](result/photo_binary70.png) |
| **T=80** | ![binary](result/photo_binary80.png) |
| **T=90** | ![binary](result/photo_binary90.png) |
| **T=100** | ![binary](result/photo_binary100.png) |


---

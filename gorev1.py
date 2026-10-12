import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Görüntüyü gri tonlamalı (grayscale) olarak yükleme
image = cv2.imread('images.jpg', cv2.IMREAD_GRAYSCALE)

# 2. Gürültüyü azaltmak için hafif Gauss filtresi (örneğin 3x3 veya 5x5 çekirdek boyutu)
blurred = cv2.GaussianBlur(image, (3, 3), sigmaX=1.0)

# 3. İstenen 3x3 dikey Prewitt çekirdeğinin (kernel) tanımlanması
prewitt_kernel_x = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)

# 4. Konvolüsyon işlemi (negatif gradyan değerlerini kaybetmemek için cv2.CV_64F kullanılır)
prewitt_edges = cv2.filter2D(blurred, cv2.CV_64F, prewitt_kernel_x)

# 5. Mutlak değer alma ve 8-bit (0-255) aralığına normalize etme
prewitt_edges_abs = np.absolute(prewitt_edges)
prewitt_edges_8u = np.uint8(np.clip(prewitt_edges_abs, 0, 255))

# Sonuçları görselleştirme
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Orijinal Gri Tonlama")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Gauss ile Yumuşatılmış")
plt.imshow(blurred, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Dikey Prewitt Filtresi Sonucu")
plt.imshow(prewitt_edges_8u, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
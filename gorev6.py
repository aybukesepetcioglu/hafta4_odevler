import cv2
import numpy as np

# 1. Görüntüyü gri oku[cite: 14]
img = cv2.imread('images.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Hata: images.jpg bulunamadı!")
else:
    # 2. 7x7 LoG matrisi
    log_kernel_7x7 = np.array([
        [ 0,  0,   1,   2,   1,  0,  0],
        [ 0,  1,   2,   0,   2,  1,  0],
        [ 1,  2,  -8, -20,  -8,  2,  1],
        [ 2,  0, -20, -40, -20,  0,  2],
        [ 1,  2,  -8, -20,  -8,  2,  1],
        [ 0,  1,   2,   0,   2,  1,  0],
        [ 0,  0,   1,   2,   1,  0,  0]
    ], dtype=np.float32)

    # ÖNEMLİ DÜZELTME 1: Matrisin toplamı 0 değilse düz alanlar patlar.
    # Bu yüzden ortalamasını çıkarıp sıfırlıyoruz:
    log_kernel_7x7 -= log_kernel_7x7.mean()

    # 3. Filtreleme (CV_64F ile yapıyoruz ki taşma olmasın)
    log_kenarlar = cv2.filter2D(img, cv2.CV_64F, log_kernel_7x7)

    # ÖNEMLİ DÜZELTME 2: Mutlak değer al ve 0-255 arasına düzgünce ölçekle (Normalize et)
    log_abs = np.absolute(log_kenarlar)
    
    # Değerleri patlatmadan 0 ile 255 arasına yayıyoruz:
    if log_abs.max() > 0:
        log_kenarlar_8u = np.uint8((log_abs / log_abs.max()) * 255)
    else:
        log_kenarlar_8u = np.uint8(log_abs)

    # 4. Göster
    cv2.imshow("Orijinal Resim", img)
    cv2.imshow("Gorev 6 - 7x7 LoG Kenarlar", log_kenarlar_8u)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
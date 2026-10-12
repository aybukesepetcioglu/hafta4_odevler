import cv2
import numpy as np

# 1. Görüntüyü gri tonlamalı oku
img = cv2.imread('images.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Hata: 'images.jpg' dosyası bulunamadı!")
else:
    # 2. GÖREV ADIMI: 3x3 Gauss filtresi ile hafif yumuşatma uygulama
    # (ksize tek sayı olmalı: (3,3), sigmaX yumuşatma şiddeti: 1.0)
    blurred = cv2.GaussianBlur(img, (3, 3), sigmaX=1.0)

    # 3. Test için 3x3 Dikey Sobel çekirdeği tanımlayalım
    kernel_x = np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]
    ], dtype=np.float32)

    # 4. Karşılaştırma:
    # A) Gauss UYGULANMAMIŞ ham görüntü üzerinde kenar tespiti
    kenar_ham = cv2.filter2D(img, cv2.CV_64F, kernel_x)
    kenar_ham_8u = np.uint8(np.clip(np.abs(kenar_ham), 0, 255))

    # B) Gauss ile YUMUŞATILMIŞ görüntü üzerinde kenar tespiti
    kenar_yumusatilmis = cv2.filter2D(blurred, cv2.CV_64F, kernel_x)
    kenar_yumusatilmis_8u = np.uint8(np.clip(np.abs(kenar_yumusatilmis), 0, 255))

    # 5. Pencerelerde gösterip gözlemleyelim
    cv2.imshow("1 - Orijinal Ham Goruntu", img)
    cv2.imshow("2 - Gauss ile Hafif Yumusatilmis", blurred)
    cv2.imshow("3 - Yumusatmasiz Kenar (Gurultulu)", kenar_ham_8u)
    cv2.imshow("4 - Gauss Sonrasi Kenar (Temiz)", kenar_yumusatilmis_8u)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
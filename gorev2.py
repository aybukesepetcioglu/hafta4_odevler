import cv2
import numpy as np

# 1. Görüntüyü gri tonlamalı (grayscale) olarak okuma
image = cv2.imread('images.jpg', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Hata: 'images.jpg' dosyası bulunamadı! Dosya adını ve yolunu kontrol edin.")
else:
    # 2. Yüksek frekanslı gürültüyü azaltmak için hafif Gauss filtresi (3x3 çekirdek)
    blurred = cv2.GaussianBlur(image, (3, 3), sigmaX=1.0)

    # 3. İstenen 3x3 yatay Prewitt çekirdeğinin tanımlanması
    prewitt_kernel_y = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=np.float32)

    # 4. Konvolüsyon işlemi (negatif gradyan değerlerini kaybetmemek için cv2.CV_64F)
    prewitt_edges = cv2.filter2D(blurred, cv2.CV_64F, prewitt_kernel_y)

    # 5. Mutlak değer alma ve 8-bit (0-255) aralığına normalize etme
    prewitt_edges_8u = np.uint8(np.clip(np.absolute(prewitt_edges), 0, 255))

    # Sonuçları ekranda gösterme
    cv2.imshow("Orijinal Gri", image)
    cv2.imshow("Gauss ile Yumusatilmis", blurred)
    cv2.imshow("Yatay Prewitt Sonucu", prewitt_edges_8u)

    # İsteğe bağlı: Sonucu diske kaydetme
    cv2.imwrite("yatay_prewitt_sonuc.png", prewitt_edges_8u)

    # Kapatmak için bir tuşa basılmasını bekle
    cv2.waitKey(0)
    cv2.destroyAllWindows()
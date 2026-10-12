import cv2
import numpy as np

# ADIM A: Görüntüyü gri tonlamalı (tek kanallı) olarak oku
# Kenar tespitinde renk bilgisine değil, ışık/parlaklık şiddetine bakarız.
img = cv2.imread('images.jpg', cv2.IMREAD_GRAYSCALE)

# Dosya okunamazsa hata almamak için kontrol edelim
if img is None:
    print("Hata: images.jpg bulunamadı! Dosya adını ve yolunu kontrol edin.")
else:
    # ADIM B: 3x3 Dikey ve Yatay Sobel matrislerimizi elle tanımlayalım
    # float32 kullanıyoruz çünkü işlem sırasında negatif sayılar çıkabilir.
    kernel_x = np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]
    ], dtype=np.float32)

    kernel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)

    # ADIM C: Konvolüsyon işlemi (filter2D)
    # cv2.CV_64F seçiyoruz: Negatif gradyanları (örn: beyazdan siyaha geçişteki -180 değerini)
    # 0'a yuvarlayıp kaybetmemek için 64-bit ondalıklı tutuyoruz.
    kenar_dikey = cv2.filter2D(img, cv2.CV_64F, kernel_x)
    kenar_yatay = cv2.filter2D(img, cv2.CV_64F, kernel_y)

    # ADIM D: Mutlak değer alma ve 8-bit'e (0-255) dönüştürme
    # Bizim için önemli olan geçişin siyah->beyaz ya da beyaz->siyah olması değil,
    # bir sınırın var olmasıdır. Bu yüzden negatif değerlerin mutlak değerini alırız.
    kenar_dikey_gorsel = np.uint8(np.clip(np.abs(kenar_dikey), 0, 255))
    kenar_yatay_gorsel = np.uint8(np.clip(np.abs(kenar_yatay), 0, 255))

    # ADIM E: Pencerelerde gösterip gözlemleyelim
    cv2.imshow("1 - Orijinal Gri Goruntu", img)
    cv2.imshow("2 - Sadece Dikey Kenarlar (X)", kenar_dikey_gorsel)
    cv2.imshow("3 - Sadece Yatay Kenarlar (Y)", kenar_yatay_gorsel)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
import cv2
import time

# 1. Fotoğrafı gri tonlamalı oku[cite: 1]
img = cv2.imread('images.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Hata: images.jpg bulunamadı!")
else:
    # 2. CANNY KENAR TESPİTİ (Çok basit tek satır)
    # 100: Alt eşik değeri, 200: Üst eşik değeri
    t0 = time.time()
    canny_kenarlar = cv2.Canny(img, 100, 200)
    canny_sure = (time.time() - t0) * 1000

    print(f"Canny Calisma Suresi: {canny_sure:.2f} ms")

    # 3. Ekranda göster
    cv2.imshow("Orijinal Resim", img)
    cv2.imshow("Canny Kenar Tespiti", canny_kenarlar)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('wall_street.jpg', cv2.IMREAD_GRAYSCALE)
def gurutu_bastirma(resim):
    gauss_kernel = np.array([
        [1,  4,  7,  4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1,  4,  7,  4, 1]
    ]) / 273.0
    
    blur_resim = cv2.filter2D(resim, -1, gauss_kernel)
    return blur_resim
def sobel_gradyan(resim):
   
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    Ix = cv2.filter2D(resim, -1, Kx)
    Iy = cv2.filter2D(resim, -1, Ky)
    
    buyukluk = np.hypot(Ix, Iy)
    buyukluk = buyukluk / buyukluk.max() * 255 
    
    yon = np.arctan2(Iy, Ix) * (180 / np.pi)
    yon[yon < 0] += 180
    
    return buyukluk, yon
def non_max_suppression(buyukluk, yon):
    satir, sutun = buyukluk.shape
    nms_resim = np.zeros((satir, sutun), dtype=np.float32)

    
    return nms_resim
def hysteresis(resim, low_thresh=50, high_thresh=150):
    yuksek_kenarlar = (resim >= high_thresh)
    zayif_kenarlar = (resim >= low_thresh) & (resim < high_thresh)
    
    
    final_resim = np.zeros_like(resim)
    final_resim[yuksek_kenarlar] = 255
    
    return final_resim
blur = gurutu_bastirma(img)
buyukluk, yon = sobel_gradyan(blur)
nms = non_max_suppression(buyukluk, yon)
sonuc = hysteresis(nms)

plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1), plt.imshow(blur, cmap='gray'), plt.title('1. Gauss Blur')
plt.subplot(1, 4, 2), plt.imshow(buyukluk, cmap='gray'), plt.title('2. Sobel Gradyan')
plt.subplot(1, 4, 3), plt.imshow(nms, cmap='gray'), plt.title('3. NMS İnceltme')
plt.subplot(1, 4, 4), plt.imshow(sonuc, cmap='gray'), plt.title('4. Canny Çıktısı')

plt.show()
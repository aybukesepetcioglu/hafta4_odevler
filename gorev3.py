import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
img = cv2.imread('images.jpg', cv2.IMREAD_GRAYSCALE)

# Gaussian blur
blurred = cv2.GaussianBlur(img, (3, 3), sigmaX=1.0)

# Sobel Kernels
# Dikey Sobel (x yönünde türev / dikey kenarlar)
sobel_kernel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

# Yatay Sobel (y yönünde türev / yatay kenarlar)
sobel_kernel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

# Konvolüsyon
sobel_x = cv2.filter2D(blurred, cv2.CV_64F, sobel_kernel_x)
sobel_y = cv2.filter2D(blurred, cv2.CV_64F, sobel_kernel_y)

# Mutlak değerler
sobel_x_8u = np.uint8(np.clip(np.abs(sobel_x), 0, 255))
sobel_y_8u = np.uint8(np.clip(np.abs(sobel_y), 0, 255))

# Birleşik gradyan büyüklüğü (Magnitude)
sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_combined_8u = np.uint8(np.clip(sobel_combined, 0, 255))

# Save individual results
cv2.imwrite('sobel_x_result.png', sobel_x_8u)
cv2.imwrite('sobel_y_result.png', sobel_y_8u)
cv2.imwrite('sobel_combined_result.png', sobel_combined_8u)

# Comparison plot
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(blurred, cmap='gray')
axes[0].set_title('Gauss ile Yumusatilmis')
axes[0].axis('off')

axes[1].imshow(sobel_x_8u, cmap='gray')
axes[1].set_title('Dikey Sobel (Sobel X)')
axes[1].axis('off')

axes[2].imshow(sobel_y_8u, cmap='gray')
axes[2].set_title('Yatay Sobel (Sobel Y)')
axes[2].axis('off')

axes[3].imshow(sobel_combined_8u, cmap='gray')
axes[3].set_title('Toplam Kenar (Bileşke)')
axes[3].axis('off')

plt.tight_layout()
plt.savefig('sobel_karsilastirma.png', dpi=200, bbox_inches='tight')
plt.close()
print("Success")
import cv2
import numpy as np
import math
from skimage.metrics import structural_similarity as compute_ssim

img=cv2.imread("lena.jpg")
h1 = np.hstack([img, img])
h2 = np.hstack([img, img])
v0= np.vstack([h1, h2])

gaussian_img=cv2.imread("gauss_img.png")
sp_img=cv2.imread("noisy_img.png")

#均值滤波
gaussian_blur = cv2.blur(gaussian_img, (3, 3))
sp_blur = cv2.blur(sp_img, (3, 3))
h1 = np.hstack([gaussian_img, gaussian_blur])
h2 = np.hstack([sp_img, sp_blur])
v1 = np.vstack([h1, h2])
cv2.imwrite(r"average_filter.jpg", v1)

#中值滤波
gaussian_blur = cv2.medianBlur(gaussian_img, 3)
sp_blur = cv2.medianBlur(sp_img, 3)
h1 = np.hstack([gaussian_img, gaussian_blur])
h2 = np.hstack([sp_img, sp_blur])
v2 = np.vstack([h1, h2])
cv2.imwrite(r"median_filter.jpg", v2)

v0[v0< 0] = 0
v0[v0 > 255] = 255
v0 = np.uint8(v0)

v1[v1< 0] = 0
v1[v1 > 255] = 255
v1 = np.uint8(v1)

v2[v2< 0] = 0
v2[v2 > 255] = 255
v2 = np.uint8(v2)

# PSNR and SSIM
def PSNR_SSIM(rec_img_rgb,orig_img):
    err_img = abs(np.array(rec_img_rgb, dtype=float) - np.array(orig_img, dtype=float))
    mse = (err_img ** 2).mean()
    psnr = 10 * math.log10((255 ** 2) / mse)
    ssim = compute_ssim(cv2.cvtColor(np.float32(rec_img_rgb), code=cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(np.float32(orig_img), code=cv2.COLOR_BGR2GRAY),data_range=255)
    return psnr,ssim

result1=PSNR_SSIM(v1,v0)
result2=PSNR_SSIM(v2,v0)
print(result1)
print(result2)

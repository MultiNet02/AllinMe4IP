import numpy as np
import pywt
from skimage import img_as_float
import matplotlib.pylab as plt
from skimage.io import imread

image = img_as_float(imread('lena.jpg'))
noise_sigma = 0.25 #16.0
image += np.random.normal(0, noise_sigma, size=image.shape)

wavelet = pywt.Wavelet('haar')
levels = int(np.floor(np.log2(image.shape[0])))
print(levels)
wavelet_coeffs = pywt.wavedec2(image, wavelet, level=levels)
# 7

def denoise(image, wavelet, noise_sigma):
    levels = int(np.floor(np.log2(image.shape[0])))
    wc = pywt.wavedec2(image, wavelet, level=levels)
    arr, coeff_slices = pywt.coeffs_to_array(wc)
    arr = pywt.threshold(arr, noise_sigma, mode='soft')
    nwc = pywt.array_to_coeffs(arr, coeff_slices, output_format='wavedec2')
    return pywt.waverec2(nwc, wavelet)

print(pywt.wavelist(kind='discrete'))
wlts = ['bior1.5', 'coif5', 'db6', 'dmey', 'haar', 'rbio2.8', 'sym15'] # pywt.wavelist(kind='discrete')
Denoised={}
for wlt in wlts:
    out = image.copy()
    for i in range(3):
        out[..., i] = denoise(image[..., i], wavelet=wlt, noise_sigma=3 / 2 * noise_sigma)
    Denoised[wlt] = np.clip(out, 0, 1)
print(len(Denoised))

plt.figure(figsize=(15,8))
plt.subplots_adjust(0,0,1,0.9,0.05,0.07)
plt.subplot(241), plt.imshow(np.clip(image,0,1)), plt.axis('off'), plt.title('original image', size=8)
i = 2
for wlt in Denoised:
    plt.subplot(2,4,i), plt.imshow(Denoised[wlt]), plt.axis('off'), plt.title(wlt, size=8)
    i += 1
plt.suptitle('Image Denoising with Wavelets', size=12)
plt.show()


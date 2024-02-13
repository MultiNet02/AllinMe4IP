import cv2
import numpy as np
from numpy import r_
from numpy import pi
import matplotlib.pyplot as plt
import time
import scipy

#统计运行时间
start = time.time()

#读取图片的灰色图像
img0=cv2.imread('lena.jpg')
#RGB-->YCbCr,保留Y分量图像
img1=cv2.cvtColor(img0, cv2.COLOR_BGR2YCR_CB)
# Cb分量赋值为0
img1[:, :, 1] = 0
# Cr分量赋值为0
img1[:, :, 2] = 0
# 重新转成rgb图像
img =cv2.cvtColor(img1, cv2.COLOR_YCR_CB2BGR)
im = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)

def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

imsize = im.shape
dct = np.zeros(imsize)
# Do 8x8 DCT on image (in-place)
for i in r_[:imsize[0]:8]:
    for j in r_[:imsize[1]:8]:
        dct[i:(i+8),j:(j+8)] = dct2( im[i:(i+8),j:(j+8)] )
end = time.time()#统计运行时间

pos = 128

# Extract a block from image
plt.figure()
plt.imshow(im[pos:pos+8,pos:pos+8],cmap='gray')
plt.title( "An 8x8 Image block")
plt.show()
# Display the dct of that block
plt.figure()
plt.imshow(dct[pos:pos+8,pos:pos+8],cmap='gray',vmax= np.max(dct)*0.01,vmin = 0, extent=[0,pi,pi,0])
plt.title( "An 8x8 DCT block")
plt.show()

print('程序执行时间: ',end - start)

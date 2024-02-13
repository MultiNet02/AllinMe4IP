import cv2
import numpy as np
from numpy import r_
from numpy import pi
import matplotlib.pyplot as plt
import time

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

imsize = im.shape
dft = np.zeros(imsize,dtype='complex')
im_dft = np.zeros(imsize,dtype='complex')
# 8x8 DFT
for i in r_[:imsize[0]:8]:
    for j in r_[:imsize[1]:8]:
        dft[i:(i+8),j:(j+8)] = np.fft.fft2( im[i:(i+8),j:(j+8)] )
end = time.time()#统计运行时间
pos = 128

# Display the dct of that block
plt.figure()
plt.imshow(abs(dft[pos:pos+8,pos:pos+8]),cmap='gray',vmax= np.max(abs(dft))*0.01,vmin = 0, extent=[0,2*pi,2*pi,0])
plt.title( "A 8x8 DFT block")
plt.show()

print('程序执行时间: ',end - start)

# 分离幅度谱与相位谱
#dft函数的输出结果是双通道的，第一个参数是结果的实数部分，第二个是结果的虚数部分
img= cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
dft=cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift=np.fft.fftshift(dft)
# 幅度谱
magnitude_spectrum = np.abs(dft_shift)
# 相位谱
phase_spectrum = np.angle(dft_shift)

#cv2的逆dft
rows,cols=img.shape
crow,ccol=int(rows/2),int(cols/2)
#创建一个有固定格式的数组
mask=np.zeros((rows,cols,2),np.uint8)
#中心上取30，下取30，左取30，右取30，设置为1
#与高频对应的部分设置为1
mask[crow-30:crow+30,ccol-30:ccol+30]=1
#与低频对应的地区设置为0
fshift=dft_shift*mask
#将低频区域转移到中间位置
f_ishift=np.fft.ifftshift(fshift)
img_back=cv2.idft(f_ishift)
#使用cv2.magnitude将实部和虚部投影到空间域，将实部和虚部转换为实部
img_back=cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
plt.imshow(img_back)
plt.title('幅度谱逆变换')
plt.axis('off')
#设置子图默认的间距
plt.tight_layout()
# 显示图像
plt.show()

import cv2
import numpy as np

#读取图片
image = cv2.imread("lena.jpg")

#设置高斯分布的均值和方差
mean = 0
#设置高斯分布的标准差
sigma = 25
#根据均值和标准差生成符合高斯分布的噪声
gauss = np.random.normal(mean,sigma,image.shape)
#给图片添加高斯噪声
noisy_img = image + gauss
#设置图片添加高斯噪声之后的像素值的范围
noisy_img = np.clip(noisy_img,a_min=0,a_max=255)
cv2.imwrite("gauss_img.png",noisy_img)

#设置添加椒盐噪声的数目比例
s_vs_p = 0.5
#设置添加噪声图像像素的数目
amount = 0.04
noise_img = np.copy(noisy_img)
#添加salt噪声
num_salt = np.ceil(amount * image.size * s_vs_p)
#设置添加噪声的坐标位置
coords = [np.random.randint(0,i - 1, int(num_salt)) for i in image.shape]
noise_img[coords[0],coords[1],:] = [255,255,255]
#添加pepper噪声
num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
#设置添加噪声的坐标位置
coords = [np.random.randint(0,i - 1, int(num_pepper)) for i in image.shape]
noise_img[coords[0],coords[1],:] = [0,0,0]
#保存图片
cv2.imwrite("noise_img.png",noise_img)

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('lena.jpg')
lenna_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 灰度化处理图像
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Canny算子
c1 = cv2.Canny(grayImage, 32, 128)

# Sobel算子
x = cv2.Sobel(grayImage, cv2.CV_16S, 1, 0)  # 对x求一阶导，垂直检测
y = cv2.Sobel(grayImage, cv2.CV_16S, 0, 1)  # 对y求一阶导，水平检测
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

#显示图形
titles = [u'canny',u'sobel']
images = [c1, Sobel]
for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

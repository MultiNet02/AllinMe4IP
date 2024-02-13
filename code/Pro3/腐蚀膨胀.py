import cv2
import numpy as np


ori = cv2.imread(r"lena.jpg")   # 读取图像

kernel1 = np.ones((3, 3), np.uint8)     # 3个不同尺度的腐蚀单元
kernel2 = np.ones((5, 5), np.uint8)
kernel3 = np.ones((7, 7), np.uint8)

erosion1 = cv2.erode(ori, kernel1)		# 腐蚀函数
erosion2 = cv2.erode(ori, kernel2)
erosion3 = cv2.erode(ori, kernel3)

dilation1 = cv2.dilate(ori, kernel1)    # 膨胀函数
dilation2 = cv2.dilate(ori, kernel2)
dilation3 = cv2.dilate(ori, kernel3)

cv2.imshow("original", ori)
cv2.imshow("erosion1", erosion1)
cv2.imshow("erosion2", erosion2)
cv2.imshow("erosion3", erosion3)
cv2.imshow("dilation1", dilation1)
cv2.imshow("dilation2", dilation2)
cv2.imshow("dilation3", dilation3)
cv2.waitKey()

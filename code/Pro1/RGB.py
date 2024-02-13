import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("lena.jpg")
# img = img[0:500,0:2000]
b, g, r = cv2.split(img)
img1 = img.copy()
img1[:, :, 0] = 0
img1[:, :, 1] = 0
cv2.imshow("R", img1)
img2 = img.copy()
img2[:, :, 0] = 0
img2[:, :, 2] = 0
cv2.imshow("G", img2)
img3 = img.copy()
img3[:, :, 1] = 0
img3[:, :, 2] = 0
cv2.imshow("B", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
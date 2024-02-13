import cv2
import matplotlib.pyplot as plt
image = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
equ = cv2.equalizeHist(image)
plt.figure("1")
plt.hist(image.ravel(), 256)
plt.figure("2")
plt.hist(equ.ravel(), 256)
plt.show()
cv2.imshow('original', image)
cv2.imshow('result', equ)
cv2.waitKey(0)
cv2.destroyAllWindows()

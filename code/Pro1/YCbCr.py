import matplotlib.pyplot as plt  # plt 用于显示图片
import numpy as np
import cv2
import copy

# RGB -> YCbCr
def rgb2ycbcr(rgb):
    m = np.array([[0.299, 0.587, 0.114],
                  [-0.1687, -0.3313, 0.5],
                  [0.5, -0.4187, -0.0813]])
    shape = rgb.shape
    if len(shape) == 3:
        rgb = rgb.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(rgb, m.transpose())

    return ycbcr.reshape(shape)


# YUV -> RGB
def ycbcr2rgb(ycbcr):
    m = np.array([[1, 0, 1.402],
                  [1, -0.344, -0.714],
                  [1, 1.772, 0]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = np.dot(ycbcr, m.transpose())
    return rgb.reshape(shape)


def main():
    # opencv的颜色通道顺序为[B,G,R]，而matplotlib颜色通道顺序为[R,G,B],所以需要调换一下通道位置
    img1 = cv2.imread('./lena.jpg')[:, :, (2, 1, 0)]  # 读取和代码处于同一目录下的 lena.jpg
    img2 = rgb2ycbcr(img1)

    # 结果展示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
    plt.subplot(221)
    # imshow()对图像进行处理，画出图像，show()进行图像显示
    plt.imshow(img2)
    plt.title('YCbCr')
    # 不显示坐标轴
    plt.axis('off')
    # print('YCbCr')
    # print(img1)

    # 子图2
    plt.subplot(222)
    img2 = rgb2ycbcr(img1)
    # Cb分量赋值为0
    img2[:, :, 1] = 0
    # Cr分量赋值为0
    img2[:, :, 2] = 0
    # 重新转成rgb图像
    img3 = ycbcr2rgb(img2)
    # print('RGB-YCbCr图像')
    # print(img2)
    img3 = img3.astype(np.uint8)
    plt.imshow(img3)
    plt.title('Y通道')
    plt.axis('off')

    # 子图3
    plt.subplot(223)
    # print('YCbCr-RGB图像')
    # print(img3)
    img2 = rgb2ycbcr(img1)
    # Y分量赋值为0
    img2[:, :, 0] = 0
    # Cr分量赋值为0
    img2[:, :, 2] = 0
    # 重新转成rgb图像
    img4 = ycbcr2rgb(img2)
    # print(img4)
    img4 = img4.astype(np.uint8)
    # print(img3)
    plt.imshow(img4)
    plt.title('Cb通道')
    plt.axis('off')

    # 子图4
    plt.subplot(224)
    img2 = rgb2ycbcr(img1)
    # Y分量赋值为0
    img2[:, :, 0] = 0
    # Cb分量赋值为0
    img2[:, :, 1] = 0
    # 重新转成rgb图像
    img5 = ycbcr2rgb(img2)
    img5 = img5.astype(np.uint8)
    plt.imshow(img5)
    plt.title('Cr通道')
    plt.axis('off')

    # #设置子图默认的间距
    plt.tight_layout()
    # 显示图像
    plt.show()


if __name__ == '__main__':
    main()
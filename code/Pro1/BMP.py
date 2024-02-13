import sys

def i2b(number, length, byteorder='little'):
    return number.to_bytes(length, byteorder)


def b2i(mbytes, byteorder='little'):
    return int.from_bytes(mbytes, byteorder)


class BMPReader:

    def __init__(self):
        self.img = None

    def read_bmp(self, filename):
        try:
            with open(filename, "rb") as file:
                # header = file.read(54)
                # BmpFileHeader
                self.bfType = file.read(2) # 0x4d42 对应BM
                self.bfSize = file.read(4) # 位图文件大小
                self.bfReserved1 = file.read(2) # 保留字段,0
                self.bfReserved2 = file.read(2) # 保留字段,0
                self.bfOffBits = file.read(4) # 偏移量
                # BmpStructHeader
                self.biSize = file.read(4) # 所需要的字节数:40
                self.biWidth = file.read(4) # 图像的宽度 单位 像素
                self.biHeight = file.read(4) # 图像的高度 单位 像素
                self.biPlanes = file.read(2) # 说明颜色平面数 总设为 1
                self.biBitCount = file.read(2)# 说明比特数
                # pixel size
                self.biCompression = file.read(4)# 图像压缩的数据类型
                self.biSizeImage = file.read(4)# 图像大小
                self.biXPelsPerMeter = file.read(4)# 水平分辨率
                self.biYPelsPerMeter = file.read(4)# 垂直分辨率
                self.biClrUsed = file.read(4)# 实际使用的彩色表中的颜色索引数
                self.biClrImportant = file.read(4)# 对图像显示有重要影响的颜色索引的数目

                print("bfType: ", b2i(self.bfType))
                print("bfSize: ", b2i(self.bfSize))
                print("biSize: ", b2i(self.biSize))
                print("biWidth: ", b2i(self.biWidth))
                print("biHeight: ", b2i(self.biHeight))
                print("biBitCount: ", b2i(self.biBitCount))

                # 读取图像数据
                image_data = file.read()
                # 转化为列表
                image = list(image_data)

                self.img = image
                return 0
        except  Exception as e:
            print(e)
            return -1

    def save_bmp(self, filename):
        # 读取文件头
        with open(filename, 'wb') as file:
            file.write(self.bfType)
            file.write(self.bfSize)
            file.write(self.bfReserved1)
            file.write(self.bfReserved2)
            file.write(self.bfOffBits)
            # reconstruct bmp header
            file.write(self.biSize)
            file.write(self.biWidth)
            file.write(self.biHeight)
            file.write(self.biPlanes)
            file.write(self.biBitCount)
            file.write(self.biCompression)
            file.write(self.biSizeImage)
            file.write(self.biXPelsPerMeter)
            file.write(self.biYPelsPerMeter)
            file.write(self.biClrUsed)
            file.write(self.biClrImportant)

            # reconstruct pixels
            file.write(bytes(self.img))
            file.close()

if __name__ == '__main__':
    br = BMPReader()
    br.read_bmp('lena.bmp')
    br.save_bmp('save.bmp')


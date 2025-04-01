import cv2
import numpy as np

def process_output(image):
    # 将图像转换为灰度图像
    gray_image = image
    _, binary_image = cv2.threshold(gray_image, 255*0.5, 255, cv2.THRESH_BINARY)

    # 定义一个 3x3 的全1矩阵作为腐蚀和膨胀操作的核
    kernel = np.ones((3,3), np.uint8)

    # 对二值图像进行开运算，去除噪声和小的对象
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    # 返回经过处理后的图像
    return cleaned_image
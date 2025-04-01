import cv2

# 读取彩色图像
image = cv2.imread('D:\pythonProjects\IML-ViT-main\IML-ViT-main\images\sample_iml_dataset\Tp\Taylor.png')

# 将彩色图像转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray.dtype)
_, binary = cv2.threshold(gray, 255 * 0.5, 255, cv2.THRESH_BINARY)
# 保存灰度图片
cv2.imwrite("D:\pythonProjects\IML-ViT-main\IML-ViT-main\images\sample_iml_dataset\Gt\Taylor.png", binary)
# 显示灰度图像
cv2.imshow('Gray Image', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

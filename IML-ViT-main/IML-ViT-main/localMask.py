# Pre-import and set for a clear ploting
from matplotlib import pyplot as plt
plt.rcParams['figure.dpi'] = 60
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (24, 6)
import sys
sys.path.append('/root/IML-ViT-main/IML-ViT-main/')
from utils.datasets import mani_dataset # special Dataset class
import cv2
import numpy as np
import sys
from PIL import Image
import os
import torch
import svgwrite
import cairosvg
from iml_vit_model import iml_vit_model

class LocalMask():
    def __init__(self):
        # 创建子进程的时候会创建类的实例，此时模型也会初始化
        self.model,self.device=self.init_model()
    
    def init_model(self):
        #加载权重信息
        # Check if GPU is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # device = 'cpu' # if out of memory, then uncommenting this line
        #print("?:",torch.cuda.is_available())
        #print("Device:", device)
        #print("PyTorch 版本:", torch.__version__)
        ckpt_path = "/root/IML-ViT-main/IML-ViT-main/checkpoints/iml-vit_checkpoint.pth"
        model = iml_vit_model()

        model.load_state_dict(
            torch.load(ckpt_path),
            strict = False
        )
        model = model.to(device)
        print('localMask模型已经初始化！')
        return model,device
    
    def localMaskImageDetect(self):
        # 定义文件夹路径
        input_folder = '/root/IML-ViT-main/IML-ViT-main/imagesFromFE/Tp'
        output_folder = '/root/IML-ViT-main/IML-ViT-main/imagesFromFE/Gt'
        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)
        # 遍历输入文件夹中的所有图片文件
        for filename in os.listdir(input_folder):
            # 构建输入和输出文件路径
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            #print(input_path)
            #print(output_path)
            # 读取彩色图像
            origin_image = cv2.imread(input_path)

            # 将彩色图像转换为灰度图像
            gray = cv2.cvtColor(origin_image, cv2.COLOR_BGR2GRAY)

            # 保存灰度图像
            cv2.imwrite(output_path, gray)

        # here we use mani dataset as example
        input_dir = "/root/IML-ViT-main/IML-ViT-main/imagesFromFE/"
        dataset = mani_dataset(
            path = input_dir,
            edge_width=7, # specify the edge mask, other wise only return 2 objects
            if_return_shape=True
        )
        # print(dataset)
        print("读取图片数量:", len(dataset))

        # #加载权重信息
        # # Check if GPU is available
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # # device = 'cpu' # if out of memory, then uncommenting this line
        # #print("?:",torch.cuda.is_available())
        # print("Device:", device)
        # #print("PyTorch 版本:", torch.__version__)
        # ckpt_path = "/root/IML-ViT-main/IML-ViT-main/checkpoints/iml-vit_checkpoint.pth"
        # model = iml_vit_model()

        # model.load_state_dict(
        #     torch.load(ckpt_path),
        #     strict = False
        # )
        # model = model.to(device)

        model=self.model
        device=self.device
        model.eval()
        outputs=[]
        with torch.no_grad():
            for img, gt, edge_mask, shape in dataset:    # Inference don't need edge mask.
                img, gt,edge_mask = img.to(device), gt.to(device),edge_mask.to(device)
                # Since no Dataloader, manually create a Batch with size==1
                img = img.unsqueeze(0) # CHW -> 1CHW
                gt = gt.unsqueeze(0)
                edge_mask = edge_mask.unsqueeze(0)
                # print("shape:", shape)
                # inference 使用模型执行测试图片
                predict_loss, mask_pred,edge_loss = model(img, gt, edge_mask)
                #print(f"Predict Loss:{predict_loss}, including edge loss: {edge_loss}")
                output = mask_pred
                #print(output)
                # Cut the origin area from padded image
                output = output[0, :, 0:shape[0], 0:shape[1]].permute(1, 2, 0).cpu().numpy()
                outputs.append(output)
                '''
                plt.title("Cropped region")
                plt.imshow(output, cmap='gray')
                plt.axis(False)
                plt.show()  # 显示图像
                '''
        # print("Done!")

        if len(outputs)==0:
            print('无待检测图片')
            sys.exit()

        
        # 这里只针对单张图片进行检测
        output=outputs[0]
        output_uint8 = (output * 255).astype('uint8')
        #print(output_uint8.dtype)

        _, binary_image = cv2.threshold(output_uint8, 255*0.3, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        # 将cv2的bgr格式转为rgb格式的图片
        # cleaned_image_rgb = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2RGB)

        # 多个mask仅保留面积最大的mask
        # 只取大于面积阈值的轮廓
        # 定义轮廓面积阈值：100可能不合理，注意调试
        #------------------------------------------------------------------------------------------------
        min_contour_area = 100  # 可以根据需求调整
        # 查找轮廓
        contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 存储符合条件的轮廓
        valid_contours = []
        # 循环遍历所有轮廓
        for contour in contours:
            # 计算轮廓的面积
            area = cv2.contourArea(contour)
            # 如果轮廓面积大于阈值，则将其添加到有效轮廓列表中
            if area > min_contour_area:
                valid_contours.append(contour)

        # 将有效轮廓存储在contours变量中
        # 现在contours中存储的是面积大于阈值的轮廓
        contours = valid_contours

        # 如果轮廓数量大于1，取前两个最大的轮廓，否则只取唯一的轮廓
        if len(contours)== 0:
            # 说明图片为真
            return 0
        elif len(contours) > 1:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]  # 取前两个最大的轮廓
        else:
            contours = contours[:1]  # 只取唯一的轮廓

        # 创建SVG文件
        height, width = origin_image.shape[:2]
        svg_filename = "/root/IML-ViT-main/IML-ViT-main/maskResults/mask.svg"
        svg = svgwrite.Drawing(svg_filename, profile='full', size=(width, height),stroke_width=2)

        # 将轮廓添加到SVG中
        for contour in contours:
            points = [f"{point[0][0]},{point[0][1]}" for point in contour]
            points = [tuple(map(int, point.split(','))) for point in points]
            svg.add(svg.polyline(points, stroke="blue", fill="none")) 

        # 保存SVG文件
        svg.save()
        return 1
    
if __name__ == "__main__":
    localMask=LocalMask()
    localMask.localMaskImageDetect()
    print("没出错！")
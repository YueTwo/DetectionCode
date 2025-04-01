import sys

import numpy as np
import os
import random
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
from demo_load_data import DatasetInstance

from templates import get_templates


class LocalFace():
    def __init__(self):
        self.MODEL_DIR = '/root/HDIL/models/'
        self.BACKBONE = 'xcp'
        self.MAPTYPE = 'tmp'
        self.LAST_EPOCH = 60
        self.MAX_EPOCHS = 60
        self.CONFIGS={
            'xcp': {
                'img_size': (299, 299),
                'map_size': (19, 19),
                'norms': [[0.5] * 3, [0.5] * 3]
            }
        }
        self.model=self.init_model()

    def get_dataset(self):
        CONFIG=self.CONFIGS[self.BACKBONE]
        #print("CONFIG:",CONFIG)
        return DatasetInstance(CONFIG['img_size'], CONFIG['map_size'], CONFIG['norms'])
    
    def init_model(self):
        from PMG_Xception import Model

        torch.backends.deterministic = True
        SEED = 1
        random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        TEMPLATES = None
        if self.MAPTYPE in ['tmp', 'pca_tmp']:
            TEMPLATES = get_templates()
        #/root/HDIL/models/tmp/models/000060.tar
        MODEL_NAME = '{0}'.format(self.MAPTYPE)
        MODEL_DIR = self.MODEL_DIR + MODEL_NAME + '/'

        resultdir = '{0}results'.format(MODEL_DIR)
        if not os.path.exists(resultdir):
            os.makedirs(resultdir)

        MODEL = Model(self.MAPTYPE, TEMPLATES, 2, False)
        MODEL.model.cuda()

        e=self.MAX_EPOCHS
        # print("使用epoch: {} 进行检测".format(e))
        # print('模型目录：',MODEL_DIR)
        MODEL.load(e, MODEL_DIR + 'models/')
        print("localFace模型已经初始化！")
        return MODEL
    
    def localFaceImageDetect(self,ossFaceImageFile):
        DATA_TEST=self.get_dataset()
        images = []

        # 以下是结果数据集
        images_paths = []
        origin_results=[]
        final_results=[]

        # 请求的携带数据
        list_res=[]

        #开始检测
        MODEL=self.model
        MODEL.model.eval()
        with torch.no_grad():
            count_images=DATA_TEST.__len__()
            if count_images == 0:
                print("未从视频中检测到人像！",)
                return []
            for index in range(count_images):
                image = DATA_TEST.__getitem__(index)
                # return {'img': img, 'im_name': im_name}
                image_path = image['im_name']
                images.append(image)

                # 图片路径列表
                images_paths.append(image_path)

                #print('图片{0}路径为:{1}'.format(index,image_path))
                img = image['img'].cuda()
                # 增加输入维度
                img = torch.unsqueeze(img, 0)
                output_2, output_3, output_concat = MODEL.model(img)
                outputs_com = output_2 + output_3 + output_concat
                
                values_list = output_concat.tolist()
                #print("原始输出：" + str(values_list))

                #原始输出列表
                origin_results.append(values_list)

                # predicted = torch.max(output_concat, dim=1)[1]
                _, predicted_com = torch.max(outputs_com.data, 1)
                final_result=predicted_com.item()

                #最终输出列表
                final_results.append(final_result)

                # 组装数据：image_path,final_result
                # 获得人脸图像名称:image_path从本地路径转为阿里云路径
                image_name = os.path.basename(image_path)
                image_path=os.path.join(ossFaceImageFile,image_name)
                obj={"address":image_path,"result":final_result}
                list_res.append(obj)
                # 0为真，1为假
                # if predicted_com:
                #     print("这是一张伪造图像")
                # else:
                #     print("这是一张真实图像")
                # #delete_image(image_path)
        # print('原始输出：')
        # print(origin_results)
        # print('检测结果：')
        # print(final_results)
        # print('人像路径：')
        # print(images_paths)
        # print('使用localFace模型检测完成！')
        print(list_res)
        return list_res

if __name__ == "__main__":
    localFace=LocalFace()
    localFace.localFaceImageDetect()
    




# def delete_image(image_path):
#     if os.path.exists(image_path):  # 检查文件是否存在
#         os.remove(image_path)  # 删除文件
#         print(f"已删除图片：{image_path}")
#     else:
#         print(f"图片不存在：{image_path}")









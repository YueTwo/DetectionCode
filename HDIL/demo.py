import sys

import numpy as np
import os
import random
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
from demo_load_data import DatasetInstance

from templates import get_templates

MODEL_DIR = '/root/HDIL/models/'
BACKBONE = 'xcp'
MAPTYPE = 'tmp'
LAST_EPOCH = 60
MAX_EPOCHS = 60

CONFIGS = {
    'xcp': {
        'img_size': (299, 299),
        'map_size': (19, 19),
        'norms': [[0.5] * 3, [0.5] * 3]
    }
}
CONFIG = CONFIGS[BACKBONE]
from PMG_Xception import Model

torch.backends.deterministic = True
SEED = 1
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def get_dataset():
    return DatasetInstance(CONFIG['img_size'], CONFIG['map_size'], CONFIG['norms'])


def delete_image(image_path):
    if os.path.exists(image_path):  # 检查文件是否存在
        os.remove(image_path)  # 删除文件
        print(f"已删除图片：{image_path}")
    else:
        print(f"图片不存在：{image_path}")


DATA_TEST = get_dataset()

TEMPLATES = None
if MAPTYPE in ['tmp', 'pca_tmp']:
    TEMPLATES = get_templates()
#/root/HDIL/models/tmp/models/000060.tar
MODEL_NAME = '{0}'.format(MAPTYPE)
MODEL_DIR = MODEL_DIR + MODEL_NAME + '/'

resultdir = '{0}results'.format(MODEL_DIR)
if not os.path.exists(resultdir):
    os.makedirs(resultdir)

MODEL = Model(MAPTYPE, TEMPLATES, 2, False)
MODEL.model.cuda()
LOSS_CSE = nn.CrossEntropyLoss().cuda()

Test_STEPS_PEREPOCH = 1

for e in range(LAST_EPOCH, MAX_EPOCHS + 1, 1):

    MODEL.load(e, MODEL_DIR + 'models/')

    MODEL.model.eval()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    all_labels_concat = []
    all_scores_concat = []
    all_labels_com = []
    all_scores_com = []

    DATA_TEST = DatasetInstance(CONFIG['img_size'], CONFIG['map_size'], CONFIG['norms'])
    images = []
    images_path = []
    origin_results=[]
    final_results=[]
    with torch.no_grad():
        count_images=DATA_TEST.__len__()
        if count_images == 0:
            print("无待处理图片")
            sys.exit()
        for index in range(count_images):
            image = DATA_TEST.__getitem__(index)
            # return {'img': img, 'im_name': im_name}
            image_path = image['im_name']

            images.append(image)
            images_path.append(image_path)

            print('图片{0}路径为:{1}'.format(index,image_path))
            img = image['img'].cuda()
            # 增加输入维度
            img = torch.unsqueeze(img, 0)
            output_2, output_3, output_concat = MODEL.model(img)
            outputs_com = output_2 + output_3 + output_concat
            '''
            print("原始输出：")
            print(output_concat)
            '''
            values_list = output_concat.tolist()
            #print("原始输出：" + str(values_list))
            origin_results.append(values_list)

            predicted = torch.max(output_concat, dim=1)[1]
            _, predicted_com = torch.max(outputs_com.data, 1)
            final_results.append(predicted_com.item())
            # 0为真，1为假
            '''
            if predicted_com:
                print("这是一张伪造图像")
            else:
                print("这是一张真实图像")
            '''
            #delete_image(image_path)
    print('原始输出：')
    print(origin_results)
    print('检测结果：')
    print(final_results)
print('检测完成')

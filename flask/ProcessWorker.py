import shutil
import time
import requests
from MyConfig import MyConfig
from OSSFileManager import OSSFileManager
from Result import Result
import torch
import os
import sys
from urllib.parse import urljoin
# 注意应该在导入模块之前添加路径
sys.path.append('/root/FakeImageDetection-master/FakeImageDetection-master/')
from universal import getVideoFrame,imageDetect
sys.path.append('/root/IML-ViT-main/IML-ViT-main/')
from localMask import LocalMask
sys.path.append('/root/HDIL/')
from localFace import LocalFace
from getFaces.utils import detect_faces_in_video

def state_response(send_to, data_to_send, attempts):
    headers = {'Content-Type': 'application/json'}
    for attempt in range(attempts):
        try:
            #print('发送的数据：', data_to_send)
            #print('发送的请求：', requests)
            response = requests.post(send_to, json=data_to_send, timeout=3,headers=headers)
            # 第一个参数是必选，且在第一个不需要指定参数名称
            # 后面几个参数需要指定参数名称
            if response.status_code == 200:
                print(f'结果发送成功，状态码: {response.status_code}')
                return
            else:
                print(f'检测结果发送失败，状态码: {response.status_code}')
        except requests.exceptions.RequestException as e:
            print(f'请求异常: {e}')
        print(f'尝试次数 {attempt + 1}/{attempts}...')
    print('请求失败，已达到最大尝试次数。')

# 在这里定义一个全局变量：保存初始化后的univeral模型，用于暴露给其他模块
# universal_init_model = None
# universal_model_device = None
def clear_folder(folder_path):
    # 检查文件夹是否为空
    if not os.listdir(folder_path):
        print("文件夹为空，无需清空:",folder_path)
        return
    else:
        # 清空文件夹
        try:
            shutil.rmtree(folder_path)
            os.mkdir(folder_path)
            print("文件夹已成功清空:",folder_path)
        except Exception as e:
            print(f"清空文件夹时出错：{e}")
    return 

def change_file_extension(file_path, new_extension):
    # 使用 os.path 模块来分割文件路径和文件名
    path, filename = os.path.split(file_path)
    # 使用 os.path 模块的 splitext 方法来分割文件名和文件扩展名
    name, _ = os.path.splitext(filename)
    # 将新的扩展名添加到文件名中
    new_file_path = os.path.join(path, name + new_extension)
    return new_file_path


def universal_worker(input_queue):
    # global universal_init_model
    # global universal_model_device
    # universal = Universal()
    # universal_init_model=universal.model
    # universal_model_device=universal.device
    # if universal_init_model is not None:
    #     print("模型已经通过func_init_model函数初始化")
    # else:
    #     print("模型没有通过func_init_model函数初始化")
    config = MyConfig()
    ossFileManager = OSSFileManager(config.AccessKey_ID, config.AccessKey_Secret, 
                                    config.endpoint, config.bucket_name,)
    fileType = None
    fileName = None
    print("universal模型子进程已开启！")

    while True:
        task = input_queue.get()  # 阻塞，直到从主进程接收到任务
        print('data:',task)
        # task是request中的json数据(在python中是字典格式)
        if task is None:
            time.sleep(0.5)
            continue
        
        try:
            # 首先清空文件夹
            clear_folder(config.universalImageFile)
            fileName = os.path.basename(task["address"])
            print("fileName:",fileName)
            ossFileManager.bucket.get_object_to_file(task["address"],os.path.join(config.universalImageFile,fileName))
            fileType=task["type"]
            print("fileType:",fileType)
            # 若为视频则先随机取一帧
            if fileType=="video":
                getVideoFrame(os.path.join(config.universalImageFile,fileName))

            result=imageDetect()
            # 释放gpu资源
            torch.cuda.empty_cache()
            # upFileName = ossFileManager.uploadFile(local_path.split("/")[-1]+fileType, local_path+fileType)
            # 发送模型产生的结果
            # 目的地址
            if fileType=="video":
                return_data={"list_res":None,"single_res":result}
                send_data = {"id":task["id"], "model":task["model"],"data":return_data}
                send_to = config.springServerIp + '/detection/video'
                state_response(send_to, send_data, config.attempts)
            else:
                return_data={"mask_svg_address":None,"result":result}
                send_data = {"id":task["id"], "model":task["model"],"data":return_data}
                send_to = config.springServerIp + '/detection/image'
                state_response(send_to, send_data, config.attempts)
        except Exception as e:
            print(f"universal检测模型出现错误, 原因:{str(e)}")


def localMask_worker(input_queue):
    config = MyConfig()
    ossFileManager = OSSFileManager(config.AccessKey_ID, config.AccessKey_Secret, 
                                    config.endpoint, config.bucket_name,)
    fileType = None
    fileName = None
    fileAddress=None
    print("localMask模型子进程已开启！")
    localMask = LocalMask()

    while True:
        task = input_queue.get()  # 阻塞，直到从主进程接收到任务
        print('data:',task)
        # task是request中的json数据(在python中是字典格式)
        if task is None:
            time.sleep(0.5)
            continue
        
        try:
            # 首先清空文件夹
            clear_folder(config.localMaskImageFile)

            fileName = os.path.basename(task["address"])
            print("fileName:",fileName)
            ossFileManager.bucket.get_object_to_file(task["address"],os.path.join(config.localMaskImageFile,fileName))
            fileType=task["type"]
            print("fileType:",fileType)
            fileAddress=task["address"]
            print("fileAddress:",fileAddress)
            
            if fileType=="video":
                print('localMask模型不可用于检测视频！')
                return
            
            result=localMask.localMaskImageDetect()
            
            # 释放gpu资源
            torch.cuda.empty_cache()
            if result:
            # 结果为0说明存在mask，图像为假
            # 上传生成的svg图像
                ossSvgFilePath=change_file_extension(fileAddress,'.svg')
                print('生成的Masksvg图像上传到：',ossSvgFilePath)
                ossFileManager.bucket.put_object_from_file(ossSvgFilePath, config.maskSvgFile)
                return_data={"mask_svg_address":ossSvgFilePath,"result":result}
                print(return_data)
                send_data = {"id":task["id"], "model":task["model"],"data":return_data}
                print(send_data)
            else:
                return_data={"mask_svg_address":None,"result":result}
                send_data = {"id":task["id"], "model":task["model"],"data":return_data}
            # 发送模型产生的结果
            # 目的地址
            send_to = config.springServerIp + '/detection/image'
            print("发送地址：",send_to)
            state_response(send_to, send_data, config.attempts)
        except Exception as e:
            print(f"localMask检测模型出现错误, 原因:{str(e)}")
            
def localFace_worker(input_queue):
    config = MyConfig()
    ossFileManager = OSSFileManager(config.AccessKey_ID, config.AccessKey_Secret, 
                                    config.endpoint, config.bucket_name,)
    fileType = None
    fileName = None
    fileAddress=None
    print("localFace模型子进程已开启！")
    localFace = LocalFace()

    while True:
        task = input_queue.get()  # 阻塞，直到从主进程接收到任务
        print('data:',task)
        # task是request中的json数据(在python中是字典格式)
        if task is None:
            time.sleep(0.5)
            continue
        
        try:
            # 首先清空文件夹
            clear_folder(config.localFaceImageFile)
            fileAddress=task["address"]
            print("fileAddress:",fileAddress)
            fileName = os.path.basename(fileAddress)
            print("fileName:",fileName)
            print("mp4文件下载到：",os.path.join(config.localFaceImageFile,fileName))
            ossFileManager.bucket.get_object_to_file(task["address"],os.path.join(config.localFaceImageFile,fileName))
            fileType=task["type"]
            print("fileType:",fileType)

            if fileType=="video":
                video_path=os.path.join(config.localFaceImageFile,fileName)
                output_folder=config.localFaceImageFile
                detect_faces_in_video(video_path,output_folder)
            # 使用localFace模型进行检测
            # 先获得阿里云上传的文件夹路径
            result=localFace.localFaceImageDetect(os.path.dirname(fileAddress))
            torch.cuda.empty_cache()

            if fileType=="image":
                singleres=result[0]['result']
                return_data={"mask_svg_address":None,"result":singleres}
                # print(return_data)
                send_data = {"id":task["id"], "model":task["model"],"data":return_data}
                print("send_data:",send_data)
                send_to = config.springServerIp + '/detection/image'
                print("图片检测结果请求路由：",send_to)
                state_response(send_to, send_data, config.attempts)
            else:
                torch.cuda.empty_cache()
                list_res=result
                # 当视频中确实检查到人脸时
                if len(list_res):
                    # 上传检测到的人脸
                    ossVideoFileFolder=os.path.dirname(fileAddress)
                    for filename in os.listdir(config.localFaceImageFile):
                        # 本地图片地址
                        img_path=os.path.join(config.localFaceImageFile,filename)
                        # oss上传地址
                        ossVideoFilePath=os.path.join(ossVideoFileFolder,filename)
                        ossFileManager.bucket.put_object_from_file(ossVideoFilePath, img_path)
                        print('人脸照片{0}已上传到{1}'.format(filename,ossVideoFilePath))
                    print('人脸照片已全部上传oss！')
                # singleres=2说明这个键值对为空，后端无需获取
                return_data={"list_res":list_res,"single_res":2}
                send_data = {"id":task["id"], "model":task["model"],"data":return_data}
                print("send_data:",send_data)
                send_to = config.springServerIp + '/detection/video'
                print("视频检测结果请求路由：",send_to)
                state_response(send_to, send_data, config.attempts)
        except Exception as e:
            print(f"localFace检测模型出现错误, 原因:{str(e)}")
import os
import cv2
import subprocess
import random

def getVideoFrame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open videofile: {video_path}")
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 随机选择一帧
    frame_number = random.randint(0, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # 读取该帧
    ret, frame = cap.read()
    if ret:
        # 保存帧到图片文件
        frame_path = os.path.splitext(video_path)[0] + ".jpg"
        cv2.imwrite(frame_path, frame)
        print(f"Saved frame to: {frame_path}")
    
    # 释放视频对象
    cap.release()
    
    # 删除mp4文件
    os.remove(video_path)
    print(f"Deleted video file: {video_path}")
    return frame_path

def imageDetect():
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(__file__)
    # 构建 test.sh 文件的完整路径
    test_script_path = os.path.join(script_dir, "test.sh")
    # 运行 shell 脚本
    subprocess.run([test_script_path, "0"])

    result_file='/root/FakeImageDetection-master/FakeImageDetection-master/predictionValues/result.txt'
    with open(result_file, 'r') as f:
        result = float(f.read())
    # 以下代码用于在控制台打印

    # percentage = result * 100
    # percentage_string = "{:.1f}%".format(percentage)
    print('universal模型原始输出：',result)  # 输出：34.5%

    if result > 0.5:
        result=1
        print('图片检测为假')
    else:
        result=0
        print('图片检测为真')
    return result
    





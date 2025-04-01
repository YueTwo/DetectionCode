#from faceImage import detect_faces
import sys
sys.path.append("/root/HDIL/")
from faceVideo import process_frames_concurrently


# def detect_faces_in_image():
#     # 图像检测
#     prototxt_path = "deploy.prototxt.txt"
#     model_path = "res10_300x300_ssd_iter_140000.caffemodel"
#     input_folder = "images"
#     output_folder = "detected_faces"
#     confidence = 0.5

#     detect_faces(prototxt_path, model_path, input_folder, output_folder, confidence)


def detect_faces_in_video(video_path,output_folder):
    # 视频检测
    # input_folder = "videos"
    # video_path = "/root/face_detector-main/face_detector-main/videos/short_video.mp4"
    output_folder = output_folder
    prototxt_path = "/root/HDIL/getFaces/deploy.prototxt.txt"
    model_path = "/root/HDIL/getFaces/res10_300x300_ssd_iter_140000.caffemodel"
    min_confidence = 0.4
    threshold = 0.05
    
    process_frames_concurrently(video_path, prototxt_path, model_path, min_confidence, threshold, output_folder)





# def clear_images(folder_path):
#     # 遍历文件夹中的所有文件
#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#         # 如果是文件而不是文件夹，并且文件名以'.jpg'、'.jpeg'、'.png'、'.bmp'等图片格式结尾
#         if os.path.isfile(file_path) and (filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg')
#                                           or filename.lower().endswith('.png') or filename.lower().endswith('.bmp')):
#             # 删除文件
#             os.remove(file_path)
#             print(f"Deleted: {file_path}")


# def main():
#     # folder_path = "/root/HDIL/imagesFromFE"
#     # clear_images(folder_path)
#     # detect_faces_in_image()
#     detect_faces_in_video()
#     # Example usage


# if __name__ == "__main__":
#     main()

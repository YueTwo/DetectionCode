import os
import uuid
import cv2
import dlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import DBSCAN


def detect_faces(frame_details, prototxt_path, model_path, min_confidence, sp, facerec, threshold):
    frame, frame_count = frame_details
    if frame_count % 2 != 0:  # 只处理每隔5帧的逻辑
        return [], []

    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    face_descriptors = []
    face_images = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > min_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dlib_image = dlib.rectangle(startX, startY, endX, endY)

            shape = sp(rgb_image, dlib_image)
            face_descriptor = facerec.compute_face_descriptor(rgb_image, shape)
            np_face_descriptor = np.array(face_descriptor)

            # 人脸图片更大一点
            # calculate expansion values
            expansion_factor = 0.1
            expandX = int((endX - startX) * expansion_factor)
            expandY = int((endY - startY) * expansion_factor)

            # adjust the coordinates to include some surrounding area
            startX = max(0, startX - expandX)
            startY = max(0, startY - expandY)
            endX = min(w, endX + expandX)
            endY = min(h, endY + expandY)

            face_images.append(frame[startY:endY, startX:endX])
            face_descriptors.append(np_face_descriptor)

    return face_descriptors, face_images


def process_frames_concurrently(video_path, prototxt_path, model_path, min_confidence, threshold, output_folder):
    cap = cv2.VideoCapture(video_path)
    sp = dlib.shape_predictor("/root/HDIL/getFaces/shape_predictor_68_face_landmarks.dat")
    facerec = dlib.face_recognition_model_v1("/root/HDIL/getFaces/dlib_face_recognition_resnet_model_v1.dat")

    frame_count = 0
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frames.append((frame, frame_count))
    cap.release()
    # 对视频取帧后删除视频文件
    if os.path.exists(video_path):
       os.remove(video_path)
       
    existing_faces = []
    face_images = []
    max_workers = 24  # 作为开始尝试的值

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(detect_faces, frame_details, prototxt_path, model_path, min_confidence, sp, facerec,
                                   threshold) for frame_details in frames]
        for future in as_completed(futures):
            face_descriptors, imgs = future.result()
            for descriptor in face_descriptors:
                is_new_face = True
                for existing_face in existing_faces:
                    distance = np.linalg.norm(descriptor - existing_face)
                    if distance < threshold:
                        is_new_face = False
                        break
                if is_new_face:
                    existing_faces.append(descriptor)
            face_images.extend(imgs)

    # DBSCAN聚类
    if existing_faces:
        clustering = DBSCAN(eps=0.5, min_samples=1, metric="euclidean").fit(existing_faces)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"Number of clusters: {n_clusters}")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for cluster_id in range(n_clusters):
            indices = [i for i, label in enumerate(labels) if label == cluster_id]
            if len(indices) == 0:
                continue
            cluster_faces = [existing_faces[index] for index in indices]
            centroid = np.mean(cluster_faces, axis=0)
            distances = [np.linalg.norm(face - centroid) for face in cluster_faces]
            representative_index = indices[np.argmin(distances)]

            representative_face_image = face_images[representative_index]
            # 获得无后缀的文件名称
            file_name_without_extension = os.path.splitext(os.path.basename(video_path))[0]

            # 唯一标识
            flag=str(uuid.uuid4())

            representative_face_path = os.path.join(output_folder, f"cluster_{cluster_id}_{file_name_without_extension}_{flag}.jpg")
            cv2.imwrite(representative_face_path, representative_face_image)
            print(f"Saved representative face for cluster {cluster_id} to: {representative_face_path}")


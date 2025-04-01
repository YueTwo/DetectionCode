import os
import numpy as np
import argparse
import cv2


def detect_faces(prototxt_path, model_path, input_folder,output_folder, confidence_threshold=0.8, expansion_factor=0.1):
    # load the serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    count = 0
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            print(f"Processing image: {image_path}")
            # load the input image and construct an input blob for the image
            image = cv2.imread(image_path)
            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))

            # pass the blob through the network and obtain the detections
            print("[INFO] computing object detections...")
            net.setInput(blob)
            detections = net.forward()

            # create directory to save detected faces if it does not exist
            output_dir = output_folder
            os.makedirs(output_dir, exist_ok=True)

            # loop over the detections
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > confidence_threshold:
                    # compute the (x, y)-coordinates of the bounding box for the
                    # object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # calculate expansion values
                    expandX = int((endX - startX) * expansion_factor)
                    expandY = int((endY - startY) * expansion_factor)

                    # adjust the coordinates to include some surrounding area
                    startX = max(0, startX - expandX)
                    startY = max(0, startY - expandY)
                    endX = min(w, endX + expandX)
                    endY = min(h, endY + expandY)

                    # extract the face ROI (region of interest)
                    face = image[startY:endY, startX:endX]

                    # construct the output path for saving the face image
                    output_path = os.path.join(output_dir, f"image_face_{count}.png")
                    count += 1

                    # save the face image
                    cv2.imwrite(output_path, face)

    # show the output image with bounding boxes
    # cv2.imshow("Output", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Example usage:
# detect_faces("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel", "path/to/input/image.jpg")

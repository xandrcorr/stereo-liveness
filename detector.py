import cv2
import numpy as np
import time

class Detector():
    def __init__(self, prototxt, caffemodel, batch_size, gpu_idx, thresh):
        self.__net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        self.__threshold = thresh

    def DetectFaces(self, image):
        (h, w) = image.shape[:2]

        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

        self.__net.setInput(blob)

        start = time.time()
        detections = self.__net.forward()
        end = time.time()
        
        # show timing information on YOLO
        # print("[INFO] YOLO took {:.6f} seconds".format(end - start))


        boxes = []
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence >= self.__threshold:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
        
                # draw the bounding box of the face along with the associated
                # probability
                y = startY - 10 if startY - 10 > 10 else startY + 10
                boxes.append([startX, startY, endX, endY])
        
        return boxes
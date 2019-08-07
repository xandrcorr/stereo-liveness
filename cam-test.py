import cv2
import numpy as np

from detector import Detector
from utils import Resize, Box, SmartCrop, exclude_face

face_detector = Detector(prototxt="Resources/detector.prototxt",
                         caffemodel="Resources/detector.caffemodel",
                         batch_size=2,
                         gpu_idx=-1,
                         thresh=0.7)

dst_size = 256

cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)

def draw_boxes(image, boxes, text):
    for box in boxes:
        cv2.rectangle(image, (box[0],box[1]), (box[2], box[3]), (255,0,0), 1)
    color = (0,255,0) if text == "Live" else (0, 0, 255)
    cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, lineType=cv2.LINE_AA) 

def crop_boxes(image, box):
    img = image[box[1]:box[3], box[0]:box[2]]
    return img

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if ret1 and ret2:
        boxes1 = face_detector.DetectFaces(frame1)
        boxes2 = face_detector.DetectFaces(frame2)
        if boxes1 and boxes2:
            exp_box1 = Box(boxes1[0][0], boxes1[0][1], boxes1[0][2]-boxes1[0][0], boxes1[0][3]-boxes1[0][1]).ExpandBox(1.5)
            exp_box2 = Box(boxes2[0][0], boxes2[0][1], boxes2[0][2]-boxes2[0][0], boxes2[0][3]-boxes2[0][1]).ExpandBox(1.5)
            sqr_box1 = exp_box1.SquareBox(1)
            sqr_box2 = exp_box2.SquareBox(1)
            crop1,_,_,_,_ = SmartCrop(frame1, sqr_box1)
            crop2,_,_,_,_ = SmartCrop(frame2, sqr_box2)

            crop1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
            crop2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)
            crop1 = Resize(crop1, dst_size, dst_size)
            crop2 = Resize(crop2, dst_size, dst_size)

            resize_coef1 = dst_size / sqr_box1.width
            resize_coef2 = dst_size / sqr_box2.width
            resized_width = int(max(resize_coef1 * exp_box1.width, resize_coef2 * exp_box2.width))

            crop1 = exclude_face(crop1, resized_width)
            crop2 = exclude_face(crop2, resized_width)

            diff = cv2.absdiff(crop1, crop2)
            diff_sum = diff.sum()
            diff_sum_norm = diff_sum / 1000000
            print(f"Diff sum: {diff_sum_norm}")
            if diff_sum_norm > 0.85:
                result = "Fake"
            else:
                result = "Live"
            draw_boxes(frame1, boxes1, result)
            draw_boxes(frame2, boxes2, result)
            cv2.imshow('cam1', frame1)
            cv2.imshow('cam2', frame2)
        cv2.waitKey(200)
    else:
        break
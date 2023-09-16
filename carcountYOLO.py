from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import tkinter as tk

#cap = cv2.VideoCapture(0) #for webcam
cap = cv2.VideoCapture("/home/fakhar/Desktop/objectDetection/CarCounter/drive2.mp4")
if not cap.isOpened():
    print("Error opening video stream or file")
#cap.set(3,1280)
#cap.set(4,720)

#defining the model
model = YOLO("../yoloweights/yolov8n.pt")

coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
    'fire hydrant', 'stop_sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donot', 'cake', 'chair', 'sofa', 
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

mask = cv2.imread('/home/fakhar/Desktop/objectDetection/CarCounter/mask2.png')

#creating the tracking instance
tracking = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

while True:
    success,img = cap.read()
    imgRegion = cv2.bitwise_and(img,mask)

    results = model(imgRegion,stream=True)
    detections = np.empty((0,5))

    #getting the bounding boxes for each of the results
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            w = x2-x1
            h = y2-y1
            #box.xywh
            #x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            #print(x1,y1,x2,y2)
            #checking the boundings

            bbox = int(x1), int(y1), int(w), int(h)
            
            x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
            #cv2.rectangle(img,(x1,y1), (x2,y2),(255,0,255), 3)
            conf = math.ceil((box.conf[0]*100))/100
            print(conf)

            

            #class names
            cls = coco_classes[int(box.cls[0])]
            if cls=='car' or cls=='truck' or cls=='bus' and conf > 0.3:

                #cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(40,y1-20)),scale=0.4,thickness=3,offset=10)

                cvzone.cornerRect(img,bbox,l=9)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))
    results = tracking.update(detections)

    for res in results:
        x1,y1,x2,y2,Id = res
        print(res)
        w,h = x2-x1,y2-y1
        cvzone.cornerRect(img,(int(x1),int(y1),int(w),int(h)),l=9,rt=2,colorR=(255,0,0))
        cvzone.putTextRect(img,f'{int(Id)}',(max(0,int(x1)),max(40,int(y1)-20)),scale=2,thickness=3 ,offset=10)
        




    cv2.imshow("Image",img)
    #cv2.imshow('imgRegion', imgRegion)
    cv2.waitKey(0)

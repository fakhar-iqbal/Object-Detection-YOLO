from ultralytics import YOLO
import cv2
import cvzone
import math

#cap = cv2.VideoCapture(0) #for webcam
cap = cv2.VideoCapture("/home/fakhar/Desktop/objectDetection/videos/drive.mp4")
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

while True:
    success,img = cap.read()
    print('success= ',success)
    results = model(img,stream=True)

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
            cvzone.cornerRect(img,bbox)
            x1,y1=int(x1), int(y1)
            #cv2.rectangle(img,(x1,y1), (x2,y2),(255,0,255), 3)
            conf = math.ceil((box.conf[0]*100))/100
            print(conf)

            

            #class names
            cls = coco_classes[int(box.cls[0])]
            cvzone.putTextRect(img,f'{cls} {conf}',(max(0,x1),max(40,y1-20)),scale=1,thickness=1)




    cv2.imshow("Image",img)
    cv2.waitKey(1)

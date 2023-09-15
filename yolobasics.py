from ultralytics import YOLO
import cv2

model = YOLO('../yoloweights/yolov8n.pt')
results = model("Images/img2.jpg",show=True)

cv2.waitKey(0)
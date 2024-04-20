import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import numpy as np


model = YOLO('YOLOv9-best.pt')  # load an official model

names = model.model.names

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
while cap.isOpened():

    success, frame = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    annotator = Annotator(frame, line_width=2)
    
    results = model.track(frame ,iou=0.5, show=False ,persist=True , tracker="bytetrack.yaml")
    
    
    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        keypoints = results[0].keypoints.xy
        boxes = results[0].boxes.xyxy.cpu()
        conf=results[0].boxes.conf.tolist() 
        
        for box, keypoint , track_id ,cof in zip(boxes , keypoints, track_ids,conf):
            if cof > 0.4 :
                annotator.draw_specific_points(keypoint,[0,1,2,3,4,5,6,7,8,9,\
                                                         10,11,12,13,14,15,16,17],[640,640],4)
                annotator.box_label(box, label=str(track_id), color=(255,0,0))
                
    # Break the loop if 'q' is pressed
    cv2.imshow('img',frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
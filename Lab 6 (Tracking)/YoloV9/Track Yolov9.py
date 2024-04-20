import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2

model = YOLO('yolov9c.pt')
names = model.model.names

cap = cv2.VideoCapture(0)

while cap.isOpened():

    success, frame = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    annotator = Annotator(frame, line_width=2)
    
    results = model.track(frame ,iou=0.5, show=False ,persist=True , tracker="bytetrack.yaml")
    
    
    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        cls=names[results[0].boxes.cls.int().cpu().tolist()[0]]
        boxes = results[0].boxes.xyxy.cpu()
        conf=results[0].boxes.conf.tolist() 
        
        for box, track_id ,cof in zip(boxes, track_ids,conf):
            # print(box.tolist())
            # print(track_id)
            if cof > 0.4 :
                annotator.box_label(box, label=str(track_id), color=(255,0,0))
                
            
    # Break the loop if 'q' is pressed
    cv2.imshow('img',frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
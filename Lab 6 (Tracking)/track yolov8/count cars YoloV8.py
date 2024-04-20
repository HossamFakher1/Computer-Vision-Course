import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2

model = YOLO('yolov8n.pt')  #yolov9c.pt / yolov8n.pt
names = model.model.names

line=((0,308),(1080,308))
cap = cv2.VideoCapture('traffictrim.mp4')

down={}


while cap.isOpened():

    success, frame = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    annotator = Annotator(frame, line_width=2)
    
    results = model.track(frame, persist=True,iou=0.5, show=False , tracker="bytetrack.yaml" )
    
    cv2.line(frame, line[0], line[1], (255,0,0),3)
    
    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        cls=names[results[0].boxes.cls.int().cpu().tolist()[0]]
        boxes = results[0].boxes.xyxy.cpu()
        conf=results[0].boxes.conf.tolist() 
        
        for box, track_id ,cof in zip(boxes, track_ids,conf):
            # print(box.int().tolist())
            # print(track_id)
            # if cof > 0.4 :
            annotator.box_label(box, label=str(track_id), color=(255,0,0))
            
            x1, y1 = int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)
            y = line[0][1] #308
            offset = 7
        
            ''' condition for red line '''
            if y < (y1 + offset) and y > (y1 - offset):
                print('#######################')
                down[track_id]=y1   
            cv2.putText(frame, f'number of cars {str(len(down))}', (100,100), cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,0,0), 3)

    # Break the loop if 'q' is pressed
    cv2.imshow('img',frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
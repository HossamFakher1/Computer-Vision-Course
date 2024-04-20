import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import time
model = YOLO('yolov9c.pt')
names = model.model.names

cap = cv2.VideoCapture('traffictrim.mp4')

cy1=322
cy2=368

offset=6

vh_down={}
counter=[]


while cap.isOpened():

    success, frame = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    annotator = Annotator(frame, line_width=2)
    
    results = model.track(frame ,persist=True ,iou=0.5, show=False ,tracker="bytetrack.yaml" ,verbose=False) #
    
    
    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        cls=names[results[0].boxes.cls.int().cpu().tolist()[0]]
        boxes = results[0].boxes.xyxy.cpu()
        conf=results[0].boxes.conf.tolist() 
        
        
        cv2.line(frame,(177,cy2),(927,cy2),(255,255,255),3)
        
        
        for box, track_id ,cof in zip(boxes, track_ids,conf):
            # print(box.tolist())
            # print(track_id)
            #if cof > 0.4 :
            annotator.box_label(box, label=str(track_id), color=(255,0,0))
            
            cx, cy = int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)
            
            
            if cy1<(cy+offset) and cy1 > (cy-offset):
               vh_down[track_id]=time.time()
               
            if track_id in vh_down:
              
               if cy2<(cy+offset) and cy2 > (cy-offset):
                 elapsed_time=time.time() - vh_down[track_id]
                 if counter.count(track_id)==0:
                    distance = 10 # meters
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6
                    counter.append([track_id,a_speed_kh])
                    
                    cv2.putText(frame,str(int(a_speed_kh))+'Km/h',(int(box[2]),int(box[1]) ),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                    
                    
            
    # Break the loop if 'q' is pressed
    cv2.imshow('img',frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
from ultralytics import YOLO

# Load the YOLOv9 model
model = YOLO('PRS Yolov9.pt')





# Open the video file
video_path = 0
cap = cv2.VideoCapture(video_path)

   
size = (1080, 720) 
video = cv2.VideoWriter('PRS.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         30, size) 
names=[
  'Paper',
  'Rock',
  'Scissors']

check={
       (0,0):'No Win -_-',
       (1,1):'No Win -_-',
       (2,2):'No Win -_-',
  
       (0,2):'Win Scissors :)',
       (2,0):'Win Scissors :)',
       
       (1,2):'Win Rock :)',
       (2,1):'Win Rock :)',
       
       (1,0):'Win Paper :)',
       (0,1):'Win Paper :)',
       }

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    frame=cv2.resize(frame, dsize=(1080,720))
    
    if success:
        # Run YOLOv8  on the frame
        results = model(frame)
        c=[]
        for result in results[0].boxes.cpu().numpy():
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            cls = int(result.cls[0])
            conf = result.conf[0].round(2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
            
            cv2.putText(frame, names[cls], (x1,y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=3, color=(255,0,0))
            
            cv2.putText(frame, str(conf), (x2,y2), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255,0,0))
            c.append(cls)
        # Visualize the results on the frame
        #annotated_frame = results[0].plot()
        if len(c)==2 :
            cv2.putText(frame, str(check[(c[0],c[1])]), (25,50), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=3, color=(0,0,255))
        # Display the annotated frame
        
        video.write(frame) 
        cv2.imshow("YOLOv9 ", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
video.release() 
cap.release()
cv2.destroyAllWindows()
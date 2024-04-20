import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('Money Yolov8.pt')



# Open the video file
video_path = 0#"http://192.168.1.9:8080/video"
cap = cv2.VideoCapture(video_path)
names = model.model.names

frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
   
size = (frame_width, frame_height) 
   
# Below VideoWriter object will create 
# a frame of above defined The output  
# is stored in 'filename.avi' file. 
output_video= cv2.VideoWriter('filenam2e.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         30, size) 



# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame=cv2.resize(frame, dsize=(640,480))

    if success:
        # Run YOLOv8  on the frame
        results = model(frame)
        
        for result in results[0].boxes.cpu().numpy():
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            cls = names[int(result.cls[0])]
            conf = result.conf[0].round(2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 5)
            
            cv2.putText(frame, cls, (x1,y1) , cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,0,0), 3)
            cv2.putText(frame, str(conf), (x2,y2), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255,0,0))
            
        # Visualize the results on the frame
        #annotated_frame = results[0].plot()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        output_video.write(frame) 
        # Display the annotated frame
        cv2.imshow("YOLOv8 ", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
output_video.release()
cv2.destroyAllWindows()
import cv2
from ultralytics import YOLOWorld

# Initialize a YOLO-World model
model = YOLOWorld('yolov8s-world.pt')  # or choose yolov8m/l-world.pt

# Define custom classes
model.set_classes(["hand"])

# Open the video file
video_path = 0
cap = cv2.VideoCapture(video_path)

names=model.model.names

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    frame=cv2.resize(frame, dsize=(640,480))

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model(frame)
        
        for result in results[0].boxes.cpu().numpy():
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            cls = names[int(result.cls[0])]
            conf = result.conf[0].round(2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
            
            cv2.putText(frame, cls, (x1,y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=3, color=(255,0,0))
            
            cv2.putText(frame, str(conf), (x2,y2), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255,0,0))
        # Visualize the results on the frame
        #annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("Yolo World", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
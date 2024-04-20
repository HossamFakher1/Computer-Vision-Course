import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp.solutions.pose.Pose()

cap=cv2.VideoCapture(0)

# cap.set(3,1280)
# cap.set(4,720)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)
result = cv2.VideoWriter('filename.avi',cv2.VideoWriter_fourcc(*'MJPG'),10, size)
while True:
    ret,image=cap.read()
    if ret==False :
        break

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    rows, cols, _ = image.shape
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks( image, results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
        
        if len(results.pose_landmarks.landmark)  == 33 :
            for id_ , lm in enumerate(results.pose_landmarks.landmark):
                cx=int(lm.x * cols)
                cy=int(lm.y * rows)
       
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    result.write(image)
    cv2.imshow('MediaPipe body', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break
        
result.release()
cap.release()
cv2.destroyAllWindows()
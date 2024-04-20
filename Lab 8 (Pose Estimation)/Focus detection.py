import numpy as np
import cv2
import mediapipe as mp
import time
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp.solutions.pose.Pose()

cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
pTime=0
x1,x2,x3,y1,y2,y3=np.inf,np.inf,np.inf,np.inf,np.inf,np.inf
while True:
    ret,image=cap.read()
    if ret==False :
        break
        
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    rows, cols, _ = image.shape
    
    if results.pose_landmarks:
        #mp_drawing.draw_landmarks( image, results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
        for id , lm in enumerate(results.pose_landmarks.landmark):
            
            cx,cy=int(lm.x*cols) , int(lm.y *rows )
            if id == 0 :
                x1,y1=cx,cy
                cv2.circle(image,(cx,cy),10,(0,0,255),-1)
            elif id == 11 :
                x2,y2=cx,cy
                cv2.circle(image,(cx,cy),10,(0,255,0),-1)
            elif id == 12 :
                x3,y3=cx,cy
                cv2.circle(image,(cx,cy),10,(255,0,0),-1)
            mid=(x2+x3)//2 
            if mid - 50 < x1 < mid + 50 and y2 - y1 > 100 :
                cv2.putText(image,"Midel",(100,100),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)
            elif x1 - x2 < x3 - x1 and y2 - y1 > 100 :
                cv2.putText(image,"Left",(100,100),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)
            elif x1 - x2 > x3 - x1 and y2 - y1 > 100 :
                cv2.putText(image,"Right",(100,100),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)
            elif y2 - y1 < 300 :
                cv2.putText(image,"Down",(100,100),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)
            
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(image,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
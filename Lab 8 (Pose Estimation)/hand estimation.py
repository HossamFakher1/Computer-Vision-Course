import numpy as np
import cv2
import mediapipe as mp
import time
mp_drawing = mp.solutions.drawing_utils
hand = mp.solutions.hands
mp_hands = mp.solutions.hands.Hands()

cap=cv2.VideoCapture(0)
# cap.set(3,1920)
# cap.set(4,1080)
pTime=0

while True:
    ret,image=cap.read()
    if ret==False :
        break
    rows, cols, _ = image.shape
    
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = mp_hands.process(image)
    image.flags.writeable = True
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks( image, hand_landmarks,hand.HAND_CONNECTIONS)
            x = [ int(landmark.x * cols) for landmark in hand_landmarks.landmark]
            y = [ int(landmark.y * rows) for landmark in hand_landmarks.landmark]
            
                
            
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(image, f'{fps:0.0f}', (10,50), cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,0,0), 3)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(10) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
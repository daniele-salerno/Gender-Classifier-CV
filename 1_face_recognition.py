###############################
"""
Part 1 of 4
Script used for recognize and testing the camera.
Click "c" to take a snapshot
Click "r" to capture frontal faces
Click "q" to quit
"""
###############################

import cv2
from datetime import datetime

rec = False
bg_mode = False
dt_mode = False
face_rec_mode = False

cap = cv2.VideoCapture(0)
codec = cv2.VideoWriter_fourcc(*'MJPG') # codec
out = None

ret, _ = cap.read()

if not ret:
    print("Webcam not availlable")
    exit(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while(cap.isOpened()):

    _, frame = cap.read()

    if(bg_mode):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if(dt_mode):
        now = datetime.now()
        str_now = now.strftime("%d/%m/%Y %H:%M:%S")
        cv2.putText(frame, str_now, (20, frame.shape[0]-20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 2)
    if(rec):
        out.write(frame)
        cv2.circle(frame, (frame.shape[1]-30, frame.shape[0]-30), 10, (0, 0, 255), cv2.FILLED)
    if(face_rec_mode):
        rects = face_cascade.detectMultiScale(frame, 1.1, 20) 
        for rect in rects:
            # draw a green rectangle around the detected face
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0,255,0), 2) 

    cv2.imshow("webcam", frame)
    k = cv2.waitKey(1)
    if(k==ord("b")):
        bg_mode = not bg_mode
        print("Black/White mode: %s" % bg_mode)
    elif(k==ord("t")):
        dt_mode = not dt_mode
        print("Show time and date: %s" % dt_mode)
    elif(k==ord("c")):
        now = datetime.now()
        filename = now.strftime("%Y%m%d%H%M%S")+".jpg"
        cv2.imwrite(filename, frame)
        print("Snapshot created: %s" % filename)
    elif(k==ord(" ")):
        if(out==None):
            out = cv2.VideoWriter('output.avi', codec, 20., (640, 480))
        rec = not rec
        print("Rec: %s" % rec)
    elif(k==ord("r")):
        face_rec_mode = not face_rec_mode
        print("Show Faces")
    elif(k==ord("q")):
        break

if(out!=None):
    # saving the video recorded
    out.release() 

cap.release()
cv2.destroyAllWindows()
import matplotlib.pyplot as plt
import cv2

print("Face Detection using openCv")

cam = cv2.VideoCapture(0)

path = r"haar cascade files/haarcascade_lefteye_2splits.xml"

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
leftEyeDetector = cv2.CascadeClassifier(path)
rightEyeDetector = cv2.CascadeClassifier("/haar cascade files/haarcascade_righteye_2splits.xml")

# "Frame" will get the next frame in the camera (via "cap").
# "Ret" will obtain return value from getting the camera frame, either true of false. 
while True:
    ret,frame = cam.read()
    if ret == False:
        continue
    faces = detector.detectMultiScale(frame,1.1,5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    leftEyes = leftEyeDetector.detectMultiScale(gray)

    # print(leftEyes)
    for leftEye in leftEyes:
        x,y,w,h = leftEye
        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)


    
    for face in faces:
        x,y,w,h = face
        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
    
    cv2.imshow('Face Detection',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
             break

cam.release()
cv2.destroyAllWindows()
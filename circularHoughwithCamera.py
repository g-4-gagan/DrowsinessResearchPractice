import cv2 
import numpy as numpy
import matplotlib.pyplot as plt

print("Iris Detection system")

cam = cv2.VideoCapture(0)

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
leftEyeDetector = cv2.CascadeClassifier("haar cascade files/haarcascade_lefteye_2splits.xml")
rightEyeDetector = cv2.CascadeClassifier("/haar cascade files/haarcascade_righteye_2splits.xml")

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
        eyeLeft = frame[y:y+h,x:x+w]
        # Convert to grayscale.
        grayEyeLeft = cv2.cvtColor(eyeLeft, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.blur(grayEyeLeft, (3, 3))
        # Apply Hough transform on the blurred image.
        detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
        	param2 = 30, minRadius = 1, maxRadius = 40)
        # Draw circles that are detected.
        if detected_circles is not None:
        	# Convert the circle parameters a, b and r to integers.
        	detected_circles = np.uint16(np.around(detected_circles))

        	print("hello")

        	for pt in detected_circles[0, :]:

        		print("yes")
        		a, b, r = pt[0], pt[1], pt[2]

        		# Draw the circumference of the circle.
        		cv2.circle(frame, (x+a, y+b), r, (0, 255, 0), 2)

        		# Draw a small circle (of radius 1) to show the center.
        		cv2.circle(frame, (x+a, y+b), 1, (0, 0, 255), 3)
    
    for face in faces:
        x,y,w,h = face
        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
    
    cv2.imshow('Face Detection',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
             break

cam.release()
cv2.destroyAllWindows()
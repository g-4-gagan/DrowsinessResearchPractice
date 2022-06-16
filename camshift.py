"""
youtube link :- https://www.youtube.com/watch?v=a9KZjQ4e6IA
code :- https://pysource.com/2018/04/09/object-tracking-with-camshift-opencv-3-4-with-python-3-tutorial-30/
"""

import cv2
import numpy as np

print("Program started")

img = cv2.imread("diary.jpg")

# cv2.imshow("Image",img)

x = 200
y = 152
width = 255-x
height = 232-y
hsv_roi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])

# cv2.imshow("HSV",roi_hist)

cam = cv2.VideoCapture(0)

term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret,frame = cam.read()

    if ret == False:
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    ret, track_window = cv2.CamShift(mask, (x, y, width, height), term_criteria)

    pts = cv2.boxPoints(ret)

    pts = np.int0(pts)

    cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
    
    
    cv2.imshow("HSV Mask", mask)
    cv2.imshow('Face Detection',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
             break

cam.release()
cv2.destroyAllWindows()

print("Program Terminated")

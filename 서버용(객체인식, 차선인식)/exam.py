import cv2
import sys
import math
import numpy as np
import RPi.GPIO as GPIO
import time

cap = cv2.VideoCapture(0)
pin =18
GPIO.setmode(GPIO.BCM)
GPIO.setup(pin, GPIO.OUT)
p= GPIO.PWM(pin, 50)
p.start(0)
cnt = 0
state = ''

while (cap.isOpened()):

    ret, src = cap.read()
    src = cv2.resize(src, (640, 360))
    src = cv2.flip(src,0)
    src = cv2.flip(src,1)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    can = cv2.Canny(gray, 50, 200, None, 3)

    height = can.shape[0]
    rectangle = np.array([[(0, height), (0, 140), (640, 140), (640, height)]])
    mask = np.zeros_like(can)
    cv2.fillPoly(mask, rectangle, 255)
    masked_image = cv2.bitwise_and(can, mask)

    ccan = cv2.cvtColor(masked_image, cv2.COLOR_GRAY2BGR)

    linesP = cv2.HoughLinesP(masked_image, 1, np.pi / 180, 50, None, 50, 30)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(ccan, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

    mimg = cv2.addWeighted(src, 1.0, ccan, 1.0, 0.0)
    cv2.imshow("original", src)
    cv2.imshow('Detect Lines',mimg)
    if linesP is not None:
        left = []
        right = []
        for i in range(0, len(linesP)):
            if linesP[i][0][0] < 320:
                left.append(linesP[i][0])
            else:
                right.append(linesP[i][0])

        left = sorted(left, key=itemgetter(1))[0]
        right = sorted(right, key=itemgetter(3))[0]

        near_l = -1
        near_r = -1

        if len(left) == 0:
            if len(right) == 0:
                pass
            else:
                near_r = right[2]
        elif len(right) == 0:
            if len(left) == 0:
                pass
            else:
                near_l = left[0]
        else:
            near_l = left[0]
            near_r = right[2]

        print(near_l, near_r)

        
        if near_l > 250:
            if state == 'right':
                pass
            else:
                state = 'right'
                p.ChangeDutyCycle(4.5)
        elif near_l < 280 | near_r > 360:
            if state == 'straight':
                pass
            else:
                state = 'straight'
                p.ChangeDutyCycle(7.5)
        elif near_r <390:
            if state == 'left':
                pass
            else:
                state = 'left'
                p.ChangeDutyCycle(10.0)
        else: pass
            


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

       
cap.release()
cv2.destroyAllWindows()


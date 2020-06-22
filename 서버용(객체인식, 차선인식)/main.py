import cv2
import sys
import math
import numpy as np
import RPi.GPIO as GPIO
import time
# import motor 
from time import sleep
from operator import itemgetter


# 모터 상태
STOP  = 0
FORWARD  = 1
BACKWORD = 2

# 모터 채널
CH1 = 0
CH2 = 1

# PIN 입출력 설정
OUTPUT = 1
INPUT = 0

# PIN 설정
HIGH = 1
LOW = 0

# 실제 핀 정의
#PWM PIN
ENA = 26  #37 pin
ENB = 0   #27 pin

#GPIO PIN
IN1 = 19  #37 pin
IN2 = 13  #35 pin
IN3 = 6   #31 pin
IN4 = 5   #29 pin
cap = cv2.VideoCapture(0)
pin =18
GPIO.setmode(GPIO.BCM)
GPIO.setup(pin, GPIO.OUT)
STOP  = 0
FORWARD  = 1
BACKWORD = 2

p= GPIO.PWM(pin, 50)
p.start(0)
cnt = 0
state = ''
# pwmA = motor.setPinConfig(ENA, IN1, IN2)
try:
    while (cap.isOpened()):

        ret, src = cap.read()
        src = cv2.resize(src, (640, 360))
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        can = cv2.Canny(gray, 50, 200, None, 3)

        height = can.shape[0]
        rectangle = np.array([[(0, height), (90, 300), (550, 300), (640, height)]])
        mask = np.zeros_like(can)
        cv2.fillPoly(mask, rectangle, 255)
        masked_image = cv2.bitwise_and(can, mask)

        ccan = cv2.cvtColor(masked_image, cv2.COLOR_GRAY2BGR)

        linesP = cv2.HoughLinesP(masked_image, 1, np.pi / 180, 50, None, 20, 30)

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
            center = []
            for i in range(0,len(linesP)):
                if linesP[i][0][0] < 320:
                    left.append(linesP[i][0][0])
                else:
                    right.append(linesP[i][0][0])
                    
            try:
                left = sorted(left, key=itemgetter(1))[0]
            except IndexError:
                pass
            try:
                right = sorted(right, key=itemgetter(3))[0]
            except IndexError:
                pass
            
    
            # print(right)

            near_l = -1
            near_r = -1            
          
            if len(left) == 0:
                if len(right) == 0:
                    pass
                else:
                    near_r = right[0]
            elif len(right) == 0:
                if len(left) == 0:
                    pass
                else:
                    near_l = left[0]
            else:
                near_l = left[0]
                near_r = right[0]
            
            print(near_l, near_r)
            
            if near_l == -1:
                print('left')
                p.ChangeDutyCycle(9.5)
            elif near_r == -1:
                print('right')
                p.ChangeDutyCycle(3)
            else:
                nl = 320-near_l
                nr = near_r-320
                if abs(nl-nr) < 170:
                    print(abs(nl-nr))
                    p.ChangeDutyCycle(7)
                else:
                    pass

#             if near_l > 250:
#                 if state == 'right':
#                     pass
#                 else:
#                     state = 'right'
#                     p.ChangeDutyCycle(4.5)
#             elif near_l < 280 | near_r > 360:
#                 if state == 'straight':
#                     pass
#                 else:
#                     state = 'straight'
#                     p.ChangeDutyCycle(7.5)
#             elif near_r <390:
#                 if state == 'left':
#                     pass
#                 else:
#                     state = 'left'
#                     p.ChangeDutyCycle(10.0)
#             else: pass

         
  
        #print(state)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            motor.setMotor(0, STOP)
            break
finally:
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()


import cv2
import sys
import math
import numpy as np

cap = cv2.VideoCapture("C:/Users/password - 1234/Downloads/video1.mp4")

while (cap.isOpened()):

    ret, src = cap.read()
    src = cv2.resize(src, (640, 360))
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
        for i in range(0,len(linesP)):
            if linesP[i][0][0] < 320:
                left.append(linesP[i][0][0])
            else:
                right.append(linesP[i][0][0])
        left = sorted(left, reverse=True)
        right = sorted(right, reverse=False)

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


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

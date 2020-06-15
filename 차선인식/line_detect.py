import cv2
import sys
import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(1)


def grayscale(src):
    return cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)


def gaussian_blur(src, kernel_size):
    return cv2.GaussianBlur(src, (kernel_size, kernel_size), 0)


def canny(src, low_t, high_t):
    return cv2.Canny(src, low_t, high_t)


def ROI(src, vert):
    mask = np.zeros_like(src)
    # print(src.shape)

    if len(src.shape) > 2:
        channel = src.shape[2]
        mask_color = (255,) * channel
    else:
        mask_color = 255

    cv2.fillPoly(mask, vert, mask_color)
    mask_src = cv2.bitwise_and(src, mask)
    return mask_src


def weighted(line, init):
    return cv2.addWeighted(src1=init, alpha=0.8, src2=line, beta=1., gamma=0.5)

# def line_detect(src, lines, color=[255,0,0], thickness=5):
#     for line in lines:
#         for x1, y1, x2, y2 in line:
#             cv2.line(src, (x1, y1), (x2, y2), color, thickness)
#
#
# def houghline(src, rho, theta, threshold, min_line_len, max_line_gap):
#     lines = cv2.HoughLinesP(src, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
#     line_img = np.zeros((src.shape[0], src.shape[1], 3), dtype=np.uint8)
#     line_detect(line_img, lines)
#     return line_img


while (True):
    ret, src = cap.read()

    src = cv2.resize(src, (640, 360))

    # kernel_size = 5 #가우시안 블러처리는 사용안했음
    dst = canny(grayscale(src), low_t=50, high_t=200)

    vert = np.array([[(50, src.shape[0]), (150, 140), (490, 140), (590, src.shape[0])]], dtype=np.int32)
    roi = ROI(dst, vert)

    ccan = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    linesP = cv2.HoughLinesP(roi, 1, np.pi / 180, 50, None, 150, 50) #hough line 출력
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(ccan, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 5, cv2.LINE_AA)

    cv2.imshow("original", ccan)
    detected = weighted(ccan, src)
    cv2.imshow('detect', detected)
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

    # plt.imshow(np.zeros_like(src))
    # plt.show()

    # linesP = cv2.HoughLinesP(roi, 1, np.pi / 180, 50, None, 50, 10)
    # if linesP is not None:
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         cv2.line(roi, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)


    # rho = 2
    # theta = np.pi / 180
    # threshold = 90
    # min_line_len = 120
    # max_line_gap = 150

    # cv2.HoughLinesP(roi, 1, np.pi / 180, 50, minLineLength=10, maxLineGap=50)
    # lines = cv2.HoughLinesP(mask, rho, theta, threshold, minLineLength=min_line_len, maxLineGap=max_line_gap)
    # lines = houghline(roi, rho, theta, threshold, min_line_len, max_line_gap)

    # dst = cv.Canny(src, 50, 200, None, 3)
    #
    # cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    # cdstP = np.copy(cdst)
    #
    # lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    #
    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    #         pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
    #         cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
    #
    # linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    #
    # if linesP is not None:
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)



    # cv.imshow("Source", src)
    # cv.imshow("Gray", dst)

    # cv.imshow("Line", lines)
    # cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    # cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

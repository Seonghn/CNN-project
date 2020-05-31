import socket
import cv2
import numpy as np



def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


HOST = '192.168.0.2'
PORT = 8080

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')
s.bind((HOST, PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')
conn, addr = s.accept()

while True:
    length = recvall(conn, 16)
    stringData = recvall(conn, int(length))
    data = np.fromstring(stringData, dtype='uint8')

    src = cv2.imdecode(data, cv2.IMREAD_COLOR)
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
    if linesP is not None:
        left = []
        right = []
        for i in range(0, len(linesP)):
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


    cv2.imshow('Detect Lines', mimg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

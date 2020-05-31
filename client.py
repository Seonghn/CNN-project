import cv2
import socket
import numpy as np
 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('220.149.236.91', 8080))

cam = cv2.VideoCapture(0)
 
cam.set(3, 320);
cam.set(4, 240);
 
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
 
while True:
    ret, frame = cam.read()
    result, frame = cv2.imencode('.jpg', frame, encode_param)
    data = np.array(frame)
    stringData = data.tostring()

    #(str(len(stringData))).encode().ljust(16)
    s.sendall((str(len(stringData))).encode().ljust(16) + stringData)
 
cam.release()

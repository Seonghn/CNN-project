import cv2
import numpy as np
import os

CONFIG = "./yolov3.cfg"  # cfg 경로
CLASSES = "./obj.names"  # classes 경로
WEIGHTS = "./yolov3.backup"  # 학습파일 경로

print(os.path.exists(CLASSES))
print(os.path.exists(CONFIG))
print(os.path.exists(WEIGHTS))


classes = None
with open(CLASSES, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

scale = 0.00392
conf_threshold = 0.01
nms_threshold = 0.01

# generate different colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# conn.send(classes[class_id].encode())

def processImage(image, index):
    Width = image.shape[1]
    Height = image.shape[0]

    # read pre-trained model and config file
    net = cv2.dnn.readNet(WEIGHTS, CONFIG)
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    # create input blob
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    # set input blob for the network
    net.setInput(blob)

    # run inference through the network
    # and gather predictions from output layers
    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    # for each detetion from each output layer
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # go through the detections remaining
    # after nms and draw bounding box
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

    # display output image
    out_image_name = "object detection" + str(index)
    # cv2.imshow(out_image_name, image)
    # wait until any key is pressed
    # cv2.waitKey()
    # save output image to disk
    cv2.imwrite("out/" + out_image_name + ".jpg", image)


index = 0
cap = cv2.VideoCapture(0)
while (cap.isOpened()):
    ret, src = cap.read()
    src = cv2.resize(src,(640, 360))
    processImage(src, index)
    index = index + 1
    cv2.imshow('ImageWindow', src)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

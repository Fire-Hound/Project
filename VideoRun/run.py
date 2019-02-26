import cv2
import numpy as np
import tensorflow as tf 
# from utils import get_output_layers, draw_bounding_box


# classes = []
# with open("classes.txt") as f:
#     classes = [line.strip() for line in f.readlines()]

# CONFIG = "big_yolo.cfg"
# WEIGHTS = "big-yolov2.weights"
# img = cv2.imread('image6.jpg')
# rows = img.shape[0]
# cols = img.shape[1]
# inp = cv2.resize(img, (300, 300))
# inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
sess = tf.Session()


new_saver = tf.train.import_meta_graph('graph/model.ckpt.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('graph/'))
ops = tf.get_default_graph().get_operations()

cap = cv2.VideoCapture("mac.mp4")
ret, frame = cap.read()

fps = 0
while (cap.isOpened()):
    ret, img = cap.read()
    frame = img[:, :, [2, 1, 0]]
    rows = frame.shape[0]
    cols = frame.shape[1]
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                    feed_dict={'image_tensor:0': frame.reshape(1, frame.shape[0], frame.shape[1], 3)})
    num_detections = int(out[0][0])
    
    for i in range(num_detections):
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]
        if score > 0.7:
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows

            cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)

    cv2.imshow('TensorFlow MobileNet-SSD', img)
    if cv2.waitKey(1) == ord('q'):
        break
sess.close()

# ret, frame = cap.read()

# fps = 0
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     #frame = frame[...,::-1]
#     Width = frame.shape[1]
#     Height = frame.shape[0]
#     #cv2.imshow('video', frame)
#     if fps%10!=0:
#         if fps>10:
#             for i in indices:
#                 i = i[0]
#                 box = boxes[i]
#                 x = box[0]
#                 y = box[1]
#                 w = box[2]
#                 h = box[3]
                
#                 draw_bounding_box(frame, classes, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
#         cv2.imshow("object detection", frame)
#         cv2.waitKey(24)
#         fps+=1
#         continue
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416,416), (0,0,0), True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(get_output_layers(net))
#     class_ids = []
#     confidences = []
#     boxes = []
#     conf_threshold = 0.5
#     nms_threshold = 0.4

#     # for each detetion from each output layer 
#     # get the confidence, class id, bounding box params
#     # and ignore weak detections (confidence < 0.5)
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.4:
#                 center_x = int(detection[0] * Width)
#                 center_y = int(detection[1] * Height)
#                 w = int(detection[2] * Width)
#                 h = int(detection[3] * Height)
#                 x = center_x - w / 2
#                 y = center_y - h / 2
#                 class_ids.append(class_id)
#                 confidences.append(float(confidence))
#                 boxes.append([x, y, w, h])

#     indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

#     # go through the detections remaining
#     # after nms and draw bounding box
#     for i in indices:
#         i = i[0]
#         box = boxes[i]
#         x = box[0]
#         y = box[1]
#         w = box[2]
#         h = box[3]
        
#         draw_bounding_box(frame, classes, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

#     # display output image    
#     cv2.imshow("object detection", frame)

#     if cv2.waitKey(1) == ord('q'):
#         break
#     fps += 1


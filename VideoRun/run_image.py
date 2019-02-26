import tensorflow as tf 
import numpy as np 
import cv2
img = cv2.imread('image1.jpg')
rows = img.shape[0]
cols = img.shape[1]
inp = cv2.resize(img, (300, 300))
inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

sess = tf.Session()


new_saver = tf.train.import_meta_graph('graph/model.ckpt.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('graph/'))
ops = tf.get_default_graph().get_operations()
out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                    feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
num_detections = int(out[0][0])

for i in range(num_detections):
    classId = int(out[3][0][i])
    score = float(out[1][0][i])
    bbox = [float(v) for v in out[2][0][i]]
    if score > 0.9:
        x = bbox[1] * cols
        y = bbox[0] * rows
        right = bbox[3] * cols
        bottom = bbox[2] * rows

        cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)

cv2.imshow('TensorFlow MobileNet-SSD', img)
cv2.waitKey()
sess.close()
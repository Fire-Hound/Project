import cv2 as cv
vcap = cv.VideoCapture("rtsp://192.168.0.2:8080/h264_pcm.sdp")
while(1):
    ret, frame = vcap.read()
    cv.imshow('VIDEO', frame)
    if cv.waitKey(1) == ord('q'):
        break
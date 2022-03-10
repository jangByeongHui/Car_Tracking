
import cv2
for i in range(6, 7):
    url = f'rtsp://admin:admin1234@218.153.209.100:502/cam/realmonitor?channel={i}&subtype=1'
    vidcap = cv2.VideoCapture(url)

    cam = 70 + i
    if vidcap.isOpened():
        ret, img = vidcap.read()
        cv2.resize(img, dsize=(720,420), interpolation=cv2.INTER_AREA)
        cv2.imwrite('./data/videos/imgSet/%d.png' % cam, img)
        cv2.imshow('frame', img)
        cv2.waitKey(1)
        print('saved image %d.png' % cam)
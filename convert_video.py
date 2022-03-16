import cv2

video_path="12.mp4"


cv2.VideoCapture(video_path)
out = cv2.VideoWriter(f'{video_path[:-4]}_convert.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (1920,1080))

while True:
    ret,frame = cap.read()
    if ret:
        out.write(frame)
    else:
        cap.release()
        out.release()
        break
    cv2.imshow("converting",frame)

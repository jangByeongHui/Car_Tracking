import cv2
import numpy as np

def main():
    test_videos = ["data/CCTV_02.mp4", "data/CCTV_10.mp4", "data/CCTV_11.mp4", "data/CCTV_12.mp4", "data/CCTV_17.mp4",
                   "data/CCTV_18.mp4", "data/CCTV_19.mp4", "data/CCTV_20.mp4", "data/CCTV_21.mp4", "data/CCTV_22.mp4",
                   "data/CCTV_23.mp4", "data/CCTV_24.mp4"]
    MAP_video="MAP.mp4"

    out = cv2.VideoWriter('all_view.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (2560,1440))
    caps=[]
    MAP_cap=cv2.VideoCapture(MAP_video)
    font = cv2.FONT_HERSHEY_SIMPLEX  # 글씨 폰트
    for test_video in test_videos:
        caps.append(cv2.VideoCapture(test_video))
    while True:
        frames=[]
        # 각 CCTV에서 이미지 가져오기
        for cap in caps:
            ret,frame = cap.read()
            if ret:
                frame=cv2.resize(frame,dsize=(640,380))
                frames.append(frame)
            else:
                Eroor_img = np.zeros((380, 640, 3), np.uint8)
                cv2.putText(Eroor_img, "Video Not Found!", (20, 70), font, 1, (0, 0, 255), 3)  # 비디오 접속 끊어짐 표시
                frames.append(Eroor_img)

        concat_frame=cv2.vconcat([cv2.hconcat(frames[0:4]),cv2.hconcat(frames[4:8]),cv2.hconcat(frames[8:12])]) # 비디오 이미지 4*3으로 합치기
        MAP_ret,MAP_frame = MAP_cap.read()

        if MAP_ret:
            MAP_frame=cv2.resize(MAP_frame,dsize=(640,380))
            MAP_frame=cv2.hconcat([MAP_frame,np.zeros((380, 640, 3), np.uint8),np.zeros((380, 640, 3), np.uint8),np.zeros((380, 640, 3), np.uint8)])
            concat_frame = cv2.vconcat([concat_frame,MAP_frame]) #지도 이미지 합치기
        else:
            MAP_cap.release()
            for i in caps:
                i.release()
                out.release()
            break
        cv2.imshow("ALL",concat_frame)
        key=cv2.waitKey(1)
        #ESC 누를 시 종료
        if key == 27:
            MAP_cap.release()
            for i in caps:
                i.release()
                out.release()
            break
        out.write(concat_frame)



if __name__ == '__main__':
    main()
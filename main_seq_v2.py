import cv2
import time
from config_hd_2 import cams
import numpy as np
import torch
import multiprocessing
import pickle

def getFrame(cctv_addr,cctv_name,img_list):
    cap = cv2.VideoCapture(cctv_addr)
    T = 0
    while True:
        ret, frame = cap.read()
        if ret:
            img_list.append((T,cctv_name,frame))
            T += 1
        else:
            break

def detect(img_list,return_points,i):
    font = cv2.FONT_HERSHEY_SIMPLEX  # 글씨 폰트
    # yolov5
    # 로컬 레포에서 모델 로드(yolov5s.pt 가중치 사용, 추후 학습후 path에 변경할 가중치 경로 입력)
    # 깃허브에서 yolov5 레포에서 모델 로드
    model = torch.hub.load('yolov5', 'custom', path='yolov5s.pt', source='local', device=i%3)
    # 검출하고자 하는 객체는 차량이기 때문에 coco data에서 검출할 객체를 차량으로만 특정(yolov5s.pt 사용시)
    model.classes = [2]
    model.conf = 0.5
    while img_list:
        T,cctv_name,img = img_list.pop()
        h, w, c = img.shape
        # 특정 구역에서만 Object 표시
        black_img = np.zeros((h, w, c), dtype=np.uint8)
        road_poly = np.array(cams[cctv_name]['road'])
        black_img = cv2.fillPoly(black_img, [road_poly], (255, 255, 255))
        black_img = cv2.cvtColor(black_img, cv2.COLOR_BGR2GRAY)
        res, thr = cv2.threshold(black_img, 127, 255, cv2.THRESH_BINARY)
        contours, his = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = contours[0]

        # 추론
        bodys = model(img, size=640)  # yolov5 추론

        # yolo5
        for i in bodys.pandas().xyxy[0].values.tolist():
            # 결과
            x1, y1, x2, y2, conf, cls, name = int(i[0]), int(i[1]), int(i[2]), int(i[3]), i[4], i[5], i[6]
            # 차량 하단 중심 좌표 표시
            target_x = int((x1 + x2) / 2)  # 차량 중심 x 좌표
            center_y = int((y1 + y2) / 2)  # 차량 중심 y 좌표
            target_y = int(y2)  # 차량 하단 y 좌표

            # 차량이 도로안에 있는 것만 검출
            if cv2.pointPolygonTest(cnt, (target_x, center_y),False) > -1:  # 도로로 표시한 polygon안에 있는 경우에만 검출 차량이 도로에 있으면 1 없으면 -1 정확히 겹치면 0
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # bounding box
                cv2.putText(img, name, (x1 - 5, y1 - 5), font, 0.5, (255, 0, 0), 1)  # class 이름
                cv2.putText(img, "{:.2f}".format(conf), (x1 + 5, y1 - 5), font, 0.5, (255, 0, 0), 1)  # 정확도

                # 보행자 픽셀 위치 표시
                img = cv2.circle(img, (target_x, target_y), 10, (255, 0, 0), -1)
                cv2.putText(img, "X:{} y:{}".format(target_x + 5, target_y + 5), (target_x + 10, target_y + 10), font,
                            0.5, (255, 0, 255), 1)

                # homography 변환
                target_point = np.array([target_x, target_y, 1], dtype=int)
                target_point.T
                H = np.array(cams[cctv_name]['homoMat'])
                target_point = H @ target_point
                target_point = target_point / target_point[2]
                target_point = list(target_point)
                target_point[0] = round(int(target_point[0]), 0)  # x - > left
                target_point[1] = round(int(target_point[1]), 0)  # y - > top
                return_points.append((T,(target_point[0],target_point[1])))

def main():
    # 작업 결과 저장 dict
    manager = multiprocessing.Manager()
    return_points = manager.list()
    img_list = manager.list()
    test_videos=["data/CCTV_02.mp4","data/CCTV_10.mp4","data/CCTV_11.mp4","data/CCTV_12.mp4","data/CCTV_17.mp4","data/CCTV_18.mp4","data/CCTV_19.mp4","data/CCTV_20.mp4","data/CCTV_21.mp4","data/CCTV_22.mp4","data/CCTV_23.mp4","data/CCTV_24.mp4"]

    #init
    work_lists=[]
    jobs=[]

    # 멀티 프로세싱할 arguments 생성
    for num,cctv_name in enumerate(cams.keys()):
        work_lists.append((test_videos[num],cctv_name,img_list))

    #프로세스 실행
    for i,work in enumerate(work_lists):
        p = multiprocessing.Process(target=getFrame, args=work)
        jobs.append(p)
        p.start()

    for i in range(3):
        p = multiprocessing.Process(target=detect,args=(img_list,return_points,i))
        jobs.append(p)
        p.start()

    #실행 완료된 프로세스 JOIN
    for proc in jobs:
        proc.join()

    with open('return_points.pkl', 'wb') as f:
        pickle.dump(return_points,f)
    print(return_points)

if __name__ == '__main__':
    main()

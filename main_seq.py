import cv2
import time
from config_hd_2 import cams
# from transmit_server import put
import numpy as np
import torch
import datetime
import multiprocessing
# import telegram

track_point=[]
new_car_index=0
Map_path = "data/videos/B3.png"
Map = cv2.imread(Map_path)

def getFrame(cctv_addr,cctv_name,return_dict):
    font = cv2.FONT_HERSHEY_SIMPLEX  # 글씨 폰트
    cap = cv2.VideoCapture(cctv_addr)
    while True:
        start_time = time.time()
        ret,frame = cap.read()
        end_time = time.time()
        #print(f'Get {cctv_name} a frame - {round(end_time - start_time, 3)} s')
        if ret:
            return_dict['img'][cctv_name] = frame
        else:
            Error_image = np.zeros((720, 1920, 3), np.uint8)
            cv2.putText(Error_image, "Video Not Found!", (20, 70), font, 1, (0, 0, 255), 3)  # 비디오 접속 끊어짐 표시
            return_dict['img'][cctv_name] = Error_image
            #retry
            cap = cv2.VideoCapture(cctv_addr)



def detect(return_dict):
    font = cv2.FONT_HERSHEY_SIMPLEX  # 글씨 폰트
    # yolov5
    # yolov5
    # 로컬 레포에서 모델 로드(yolov5s.pt 가중치 사용, 추후 학습후 path에 변경할 가중치 경로 입력)
    # 깃허브에서 yolov5 레포에서 모델 로드
    # model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt',device=num%3)
    model = torch.hub.load('yolov5', 'custom', path='yolov5s.pt', source='local', device=0)
    # 검출하고자 하는 객체는 차량이기 때문에 coco data에서 검출할 객체를 차량으로만 특정(yolov5s.pt 사용시)
    model.classes = [0,2]
    model.conf = 0.7
    window_width=320
    window_height=270
    # CCTV 화면 정렬
    for num,cctv_name in enumerate(cams.keys()):
        cv2.namedWindow(cctv_name)
        cv2.moveWindow(cctv_name,320*(num%6),270*(num//6))

    # CCTV 화면 추론
    while True:

        for cctv_name in cams.keys():
            img = return_dict['img'][cctv_name]  # 추론할 이미지
            h,w,c =img.shape

            # 특정 구역에서만 Object 표시
            black_img = np.zeros((h, w, c), dtype=np.uint8)
            road_poly=np.array(cams[cctv_name]['road'])
            black_img = cv2.fillPoly(black_img,[road_poly], (255, 255, 255))
            black_img = cv2.cvtColor(black_img, cv2.COLOR_BGR2GRAY)
            res, thr = cv2.threshold(black_img, 127, 255, cv2.THRESH_BINARY)
            contours, his = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt=contours[0]
            # 추론
            start_time=time.time()
            bodys = model(img, size=640) # yolov5 추론
            end_time=time.time()
            print(f'yolov5 {cctv_name} img 추론 시간 - {round(end_time - start_time, 3)} s')
            flag = False
            points = []

            # yolo5
            for i in bodys.pandas().xyxy[0].values.tolist():
                # 결과
                x1, y1, x2, y2, conf, cls, name = int(i[0]), int(i[1]), int(i[2]), int(i[3]), i[4], i[5], i[6]
                # 차량 하단 중심 좌표 표시
                target_x = int((x1 + x2) / 2)  # 차량 중심 x 좌표
                target_y = int(y2)  # 차량 하단 좌표

                # 차량이 도로안에 있는 것만 검출
                if cv2.pointPolygonTest(cnt,(target_x,target_y),False)>-1:  # 도로로 표시한 polygon안에 있는 경우에만 검출 차량이 도로에 있으면 1 없으면 -1 정확히 겹치면 0
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
                    points.append((target_point[0], target_point[1])) #Homography로 계산 좌표 값
                    flag = True  # 변환된 정보 저장
                #Yolov5 추론 끝

            # 변환된 보행자 픽셀 위치 저장
            if flag:
                return_dict[cctv_name] = (flag, points)
            else:
                return_dict[cctv_name] = (False, [])
            temp_img = cv2.resize(img, dsize=(window_width, window_height))
            cv2.imshow(cctv_name, temp_img)
            cv2.imwrite("runs/video/"+cctv_name+".jpg",temp_img)
        # send2server(return_dict)
        Stich_Car(return_dict)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break


def Stich_Car(data):
    global new_car_index
    global track_point
    global Map
    COLORS = [(0,0,255),(255,0,0),(0,255,0),(255,255,0),(0,255,255),(100,100,100),(255,0,255)] #표시할 색상들
    temp_points = [] # 지도에 표시할 모든 좌표를 담을 좌표

    for cctv_name in cams.keys(): # 모든 CCTV에서 좌표 가져오기
        flag,points = data[cctv_name]
        for point in points:
            temp_points.append(point) #임시 변수에 좌표들 저장

    track_point.extend(temp_points) #기존 트랙킹하는 좌표안에 임시로 저장한 좌표들 저장

    #트랙킹하는 좌표를 표시
    for num, (tx, ty) in enumerate(track_point):
        Map = cv2.circle(Map, (tx, ty), 30, COLORS[num%7], -1)  # 지도 위에 점으로 표시
    temp_Map = cv2.resize(Map, dsize=(720, 480))
    cv2.imshow("Map", temp_Map)
    cv2.imwrite("runs/MAP/result.jpg",temp_Map)




# 추후 서버 전송
# MQTT 전송시에는 데이터를 문자열로 보내야 한다.
def send2server(data):
    bot = telegram.Bot(token="5137138184:AAEf4mPnuYIz2YT5HWGACYy5cKHsgo68OPY")
    chat_id = 1930625013
    Map_path = "./data/B3.png"
    Map = cv2.imread(Map_path)
    try:
        temp_list = []
        state = False
        for cctv_name in cams.keys():
            flag, points = data[cctv_name]
            if flag:
                state = True
                for num, (x, y) in enumerate(points):
                    Map = cv2.circle(Map, (x, y), 30, (0, 255, 0), -1)  # 지도위에 표시
                    temp_list.append({'id': f'{cctv_name}_{num + 1}', 'top': y, 'left': x,
                                      'update': str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))})
                bot.sendMessage(chat_id=chat_id, text=f'cctv : {cctv_name} found {num+1} people!')
        temp_Map = cv2.resize(Map, dsize=(720, 480))
        cv2.imshow("Map", temp_Map)
        if state:
            print(f'state:{state} data:{temp_list}')
            put(f'{temp_list}')
    except Exception as e:
        print(f'state:{state} data:{temp_list}')
        print("Send2Server Error : {}".format(e))
        pass


def main():
    # 작업 결과 저장 dict
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    return_dict['img'] = manager.dict()
    test_videos=["data/videos/Anyang2_SKV1_ch1_20220121090906.mp4","data/videos/Anyang2_SKV1_ch2_20220126165051_20220126165101.mp4","data/videos/Anyang2_SKV1_ch3_20220126165125_20220126165210.mp4","data/videos/Anyang2_SKV1_ch4_20220124132217_20220124132240.mp4","data/videos/Anyang2_SKV1_ch5_20220126165037_20220126165047.mp4"]
    #init
    for cctv_name in cams.keys():
        return_dict['img'][cctv_name] = np.zeros((1080, 1920, 3), np.uint8)
    work_lists=[]
    jobs=[]

    # 멀티 프로세싱할 arguments 생성
    for num,cctv_name in enumerate( cams.keys()):
        work_lists.append((test_videos[num],cctv_name,return_dict))

    #프로세스 실행
    for i,work in enumerate(work_lists):
        p = multiprocessing.Process(target=getFrame, args=work)
        jobs.append(p)
        p.start()
    else:
        p = multiprocessing.Process(target=detect, args=(return_dict,))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

if __name__ == '__main__':
    main()

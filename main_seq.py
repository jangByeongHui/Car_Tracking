import cv2
import time
from config_hd_2 import cams
from transmit_server import put
import numpy as np
import torch
# import telegram

track_point=[[] for i in range(10)]
FRANME_SYNC=-1
new_car_index=0
Map_path = "data/videos/B3.png"
Map = cv2.imread(Map_path)
window_width = 330
window_height = 270
# yolov5
# 로컬 레포에서 모델 로드(yolov5s.pt 가중치 사용, 추후 학습후 path에 변경할 가중치 경로 입력)
# 깃허브에서 yolov5 레포에서 모델 로드
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt',device=num%3)
model = torch.hub.load('yolov5', 'custom', path='yolov5s.pt', source='local', device=0)
# 검출하고자 하는 객체는 차량이기 때문에 coco data에서 검출할 객체를 차량으로만 특정(yolov5s.pt 사용시)
model.classes = [2]
model.conf = 0.5
font = cv2.FONT_HERSHEY_SIMPLEX  # 글씨 폰트
# 표시할 색상들
COLORS = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255), (100, 100, 100), (255, 0, 255),
          (255, 69, 0), (173, 255, 47), (100, 149, 237), (148, 0, 211), (255, 105, 180), (244, 164, 96),
          (240, 255, 255)]
len_COLORS = len(COLORS)

def detect(cctv_name,img):
    global window_height,window_width,font,model
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
    bodys = model(img, size=640) # yolov5 추론
    # 현재 이미지에서 변환된 좌표를 담을 이미지
    points = []

    # yolo5
    for i in bodys.pandas().xyxy[0].values.tolist():
        # 결과
        x1, y1, x2, y2, conf, cls, name = int(i[0]), int(i[1]), int(i[2]), int(i[3]), i[4], i[5], i[6]
        # 차량 하단 중심 좌표 표시
        target_x = int((x1 + x2) / 2)  # 차량 중심 x 좌표
        center_y = int((y1 + y2) / 2)  # 차량 중심 y 좌표
        target_y = int(y2)  # 차량 하단 y 좌표

        # 차량이 도로안에 있는 것만 검출
        if cv2.pointPolygonTest(cnt,(target_x,center_y),False)>-1:  # 도로로 표시한 polygon안에 있는 경우에만 검출 차량이 도로에 있으면 1 없으면 -1 정확히 겹치면 0
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
    resize_img = cv2.resize(img, dsize(window_height, window_width))
    cv2.imshow(cctv_name, resize_img)
    return points

def finddistance(x1,y1,x2,y2):
    a = np.array((x1, y1,0))
    b = np.array((x2, y2, 0))
    return np.sqrt(np.sum(np.square(a-b)))

def Stich_Car(points):
    global new_car_index,track_point,FRANME_SYNC,Map,font,COLORS,len_COLORS
    temp_points = [] # 여러 CCTV에 의해 중복되는 좌표를 제거 후 담을 좌표들
    all_temp_points =[] # 지도에 표시할 모든 좌표를 담을 좌표들
    threshold_dist = 400 #일정 거리 이하일시 같은 좌표로 판단
    temp_trackpoints=[] # 현재 추론된 차량 트랙킹 정보를 담을 리스트

    #초기 모든 지도 죄표 저장
    for (pX,pY) in points:
        all_temp_points.append((pX,pY))

    # 추론된 좌표들 중에서 이미 같은 것이라고 판단된 좌표들은 삭제
    for (aX,aY) in all_temp_points:
        for (tX,tY) in temp_points:
            if finddistance(aX,aY,tX,tY)<threshold_dist: #설정한 거리보다 가까우면 이미 포함된 좌표라고 판단
                break
        else:
            temp_points.append((aX,aY)) #기존에 없던 새로운 좌표

    # 이전 좌표들과 최대한 가까운 좌표 검출
    for (tX,tY) in temp_points:
        Min = 1e9 # 최소값을 찾기 위한 초기화
        Min_car_index=0 # 유사성이 높은 차량 번호
        similar_flag=0 # 유사성이 높은 차량이 존재여부 FLAG

        #이전 10프레임 좌표들과 비교하였을 때 가장 비슷한 좌표 찾기 -> 이를 통해 같은 차량이라고 판단
        track_points=[]
        for num in range(9):
            track_points.extend(track_point[num])
        for (car_index,prevX,prevY) in track_points:
            dist = finddistance(tX,tY,prevX,prevY) # 이전 좌표와 현재 좌표거리들을 비교
            if dist<threshold_dist: # 좌표거리가 특정 거리 이하면은 판단
                if Min>dist: # 특정 거리 이하 중에 제일 가까운 좌표
                    similar_flag = True # 유사성이 있는 좌표가 있는 것으로 판단
                    Min=dist # 가장 유사성이 높은 좌표 거리 저장
                    Min_car_index=car_index #유사성이 높은 차량 index번호 저장

        if similar_flag:
            temp_trackpoints.append((Min_car_index,tX,tY)) #유사성이 있었으니 기존 index에 추가
        else:
            temp_trackpoints.append((new_car_index+1,tX,tY)) # 유사성 있는 것이 없었으니 새로운 값으로 할당
            new_car_index+=1 # 새로운 차량 추가

    #트랙킹하는 좌표를 표시
    save_map=False #트랙킹 하는 객체가 검출된 경우에만 저장
    for num, (car_index,tx, ty) in enumerate(temp_trackpoints):
        Map = cv2.circle(Map, (tx, ty), 30, COLORS[car_index%len_COLORS], -1)  # 지도 위에 점으로 표시
        cv2.putText(Map,str(car_index), (tx, ty - 15), font, 2, (0, 0, 0), 3)  # car_index 표시
        save_map=True
    temp_Map = cv2.resize(Map, dsize=(720, 480))
    cv2.imshow("Map", temp_Map)

    # 트랙킹한 결과 저장
    if save_map:
        now=time.localtime()
        now_TIME="%04d%02d%02d_%02d%02d%02d"%(now.tm_year,now.tm_mon,now.tm_mday,now.tm_hour,now.tm_min,now.tm_sec)
        cv2.imwrite(f'runs/detect/MAP/{now_TIME}.jpg',temp_Map)

    #최대 10개의 이전 프레임 기록을 저장
    FRANME_SYNC=(FRANME_SYNC+1)%10 #FRAME_SYNC는 0~9 값을 가지고 이전 기록을 계속해서 저장
    track_point[FRANME_SYNC]=temp_trackpoints #기존 트랙킹하는 좌표안에 임시로 저장한 좌표들 저장

# MQTT 전송시에는 데이터를 문자열로 보내야 한다.
def send2server(data):
    #bot = telegram.Bot(token="5137138184:AAEf4mPnuYIz2YT5HWGACYy5cKHsgo68OPY")
    #chat_id = 1930625013
    Map_path = "data/videos/B3.png"
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
                #bot.sendMessage(chat_id=chat_id, text=f'cctv : {cctv_name} found {num+1} people!')
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
    # 초기 설정 값
    test_videos=["data/CCTV_02.mp4","data/CCTV_10.mp4","data/CCTV_11.mp4","data/CCTV_12.mp4","data/CCTV_17.mp4","data/CCTV_18.mp4","data/CCTV_19.mp4","data/CCTV_20.mp4","data/CCTV_21.mp4","data/CCTV_22.mp4","data/CCTV_23.mp4","data/CCTV_24.mp4"]

    # CCTV 화면 정렬
    for num, cctv_name in enumerate(cams.keys()):
        cv2.namedWindow(cctv_name)
        cv2.moveWindow(cctv_name, window_width * (num % 6), window_height * (num // 6))

    #videocapture 객체 저장
    for index,cctv_name in enumerate(cams.keys()):
        cams[cctv_name]['cap']=cv2.VideoCapture(test_videos[index])

    while True:
        MAP_Points =[]
        for cctv_name in cams.keys():
            #초기 빈화면
            img = np.zeros((720, 1920, 3), np.uint8)
            cv2.putText(Error_image, "Video Not Found!", (20, 70), font, 1, (0, 0, 255), 3)  # 비디오 접속 끊어짐 표시
            ret,frame = cams[cctv_name]['cap'].read() #각각 CCTV별로 이미지 가져오기
            if ret:
                img=frame
            else:
                cams[cctv_name]['cap'].release() #videocapture 객체 프리
            new_points = detect(cctv_name,img) # 추론한 좌표 가져오기
            MAP_Points.extend(new_points) # 추론 좌표 저장
        Stich_Car(MAP_Points)




if __name__ == '__main__':
    main()

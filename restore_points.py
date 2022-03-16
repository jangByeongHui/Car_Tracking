import pickle
import cv2
import numpy as np

# 객체 구분 색상
COLORS = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255), (100, 100, 100), (255, 0, 255),
          (255, 69, 0), (173, 255, 47), (100, 149, 237), (148, 0, 211), (255, 105, 180), (244, 164, 96),
          (240, 255, 255)]
# 글씨 폰트
font = cv2.FONT_HERSHEY_SIMPLEX

# 직선 거리구하기 함수
def finddistance(x1,y1,x2,y2):
    a = np.array((x1, y1,0))
    b = np.array((x2, y2, 0))
    return np.sqrt(np.sum(np.square(a-b)))

#저장된 좌표들 가져오기
with open('return_points.pkl','rb') as f:
    list = pickle.load(f)

#지도 이미지 가져오기
MAP_PATH = "data/B3.png"
Map = cv2.imread(MAP_PATH)

# 동영상 저장시 초기 설정
# h,w,c = Map.shape
# out = cv2.VideoWriter('Tracking.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (w,h))

list.sort(reverse=True) #예전 프레임일수록 뒤에

now_frames = 0 # 현재 프레임 수 저장
now_points = []
track_points = dict()
idx = 0

while list:
    # 같은 프레임 정보만 저장
    if now_frames == list[-1][0]:
        frame,points = list.pop()
        now_points.append(points)
    else:
        new_tackpoints = []
        for nx,ny in now_points:
            for trackobject in track_points:
                id,tx,ty = trackobject
                if finddistance(nx,ny,tx,ty)<300:
                    new_tackpoints.append((id,nx,ny))
                    break
            else:
                new_tackpoints.append((idx,nx,ny))
                idx += 1
        # 지도위에 표시
        monitor_img = np.zeros((2000,1000, 3), np.uint8)
        cv2.putText(monitor_img, f'NOW FRAMES : {now_frames}', (0, 50), font, 2, (0, 250, 0), 5)
        for num,value in enumerate(new_tackpoints):
            id,x,y = value
            Map = cv2.circle(Map, (x, y), 30,COLORS[id%len(COLORS)], -1)  # 지도위에 표시
            cv2.putText(Map, f'{id}',(x-15,y+15), font,1.5, (0, 0,0), 4)
            cv2.putText(monitor_img, f'ID : {id} X : {x} Y : {y}', (0,20+100*(num+1)), font, 2, (0,250, 0), 5)
        now_frames=list[-1][0]
        track_points=new_tackpoints
        now_points =[]
        print(track_points)
        cv2.imshow("MAP",Map)
        cv2.imshow("monitor", monitor_img)
        key = cv2.waitKey(1)
        if key == 27:
            break
        if key == ord('p'):
            cv2.waitKey(-1)





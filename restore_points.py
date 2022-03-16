import pickle
import cv2

MAP_PATH = "data/B3.png"

with open('return_points.pkl','rb') as f:
    list = pickle.load(f)

list.sort()
with open('return_points.pkl','wb') as f:
    list = pickle.dump(list,f)

Map = cv2.imread(MAP_PATH)
h,w,c = Map.shape
out = cv2.VideoWriter('Tracking.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (w,h))
for i in list:
    frames,points = i
    Map = cv2.circle(Map, (points[0], points[1]), 30, (0, 255, 0), -1)  # 지도위에 표시
    out.write(Map)
    cv2.imshow("MAP",Map)
    cv2.waitKey(1)
else:
    out.release()


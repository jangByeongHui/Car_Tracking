import pickle
import cv2
MAP_PATH = "data/B3.png"
list = []

with open('points.pkl','rb') as f:
    list = pickle.load(f)

Map = cv2.imread(MAP_PATH)

for i in list:
    frames,points = i
    Map = cv2.circle(Map, (points[0], points[1]), 30, (0, 255, 0), -1)  # 지도위에 표시
    cv2.imshow("MAP",Map)
    cv2.waitKey(1)


from _typeshed import NoneType
import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import True_, random, record
import homo_point, parking_position, camera_link
import os, sys, math
from threading import Thread

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

def find_parking_position(x, y):
    # spot = api.get_parking_spot() # 한번 call할때마다 api를 불러오는 것이 비효율적이라면 parking_position.py에서 가져다 쓰는 것이 더 좋은 방법
    spot = parking_position.position
    width = 64
    height = 128
    margin = 3
    position = ''
    for key in spot:
        # top = spot[key]['top']
        # left = spot[key]['left']
        top = spot[key][0]
        left = spot[key][1]
        if left-margin <= x < left+width+margin and top-margin <= y < top+height+margin:
            position = key
            break
    return position

def find_parking_cam_num(x, y):
    height = 128 # 카메라가 보는 양옆 주차공간까지
    for cam_num in homo_point.pts_src:
        x1 = homo_point.pts_src[cam_num][1][0] # top left
        y1 = homo_point.pts_src[cam_num][1][1]
        x2 = homo_point.pts_src[cam_num][3][0]
        y2 = homo_point.pts_src[cam_num][3][1] # bottom right

        if x1 <= x < x2 and y1-height < y <= y2+height:
            return cam_num

def is_enter(x, y):
    for cam_num in homo_point.entrance:
        x1 = homo_point.entrance[cam_num][0][0]
        y1 = homo_point.entrance[cam_num][0][1]
        x2 = homo_point.entrance[cam_num][1][0]
        y2 = homo_point.entrance[cam_num][1][1]

        if x1 <= x < x2 and y1 < y <= y2:
            return True
    return False

def check_iou(x1, y1, x2, y2):
    l = 153 # car length
    d = 64 # car width
    if abs(x2-x1) < l and abs(y2-y1) < d:
        return True
    else:
        return False

def next_camera(cam):
    next_camera = []
    for key in camera_link.link:
        if key == cam:
            next_camera = camera_link.link[key]
            break
    
    return next_camera

def get_camera_index(cam, CCTV_exist):
    row, col = None, None
    for i in range(len(CCTV_exist)):
        if cam in CCTV_exist[i]:
            row = i
            col = CCTV_exist[i].index(cam)
    return row, col


def watch_cctv(cam_num, row, col, CCTV_OPEN, CCTV_exist, car_list):
    time_vector = [[]]
    rect_list = np.array(homo_point.pts_dst[cam_num])

    # Create a black image
    img_b = np.zeros((480, 720, 3), dtype=np.uint8)
    img_b = cv2.fillPoly(img_b, [rect_list], (255, 255, 255))
    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    res, thr = cv2.threshold(img_b, 127, 255, cv2.THRESH_BINARY)
    contours_b, his = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Homograpy 처리 후 매핑작업
    im_src = cv2.imread('ch_b1.png')  # 1층
    pts_src = np.array(homo_point.pts_src[cam_num])
    cv2.polylines(im_src, [pts_src], True, (0, 0, 0), 2)

    im_dst = cv2.imread('cam80.png')
    pts_dst = np.array(homo_point.pts_dst[cam_num])
    cv2.polylines(im_dst, [pts_dst], True, (255, 255, 255), 2)
    h1, status = cv2.findHomography(pts_dst, pts_src)

    while CCTV_OPEN[row][col] > 0:
        ret, im0s = cam_list[row][col].read()

        img = letterbox(im0s, imgsz, stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            t3 = time_synchronized()
            #  이미지 리사이즈
            im0 = cv2.resize(im0, (720, 480))

            p = Path(p)  # to Path-
            save_path = str(save_dir / p.name)  # img.jpg
            s += '%gx%g ' % img.shape[2:]  # print string(img shape)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            imc = im0.copy() if opt.save_crop else im0  # for opt.save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if opt.save_crop or view_img:  # Add bbox to image
                        x1, y1 = (int(xyxy[0]) + int(xyxy[2])) / 2, int(xyxy[3])  # 중심점 좌표 x, y
                        
                        dist = cv2.pointPolygonTest(contours_b[0], (x1, y1), False)
                        # if names[int(cls)] == 'car' or names[int(cls)] == 'person':  # box 가 차량이나 사람일 경우
                        if names[int(cls)] == 'car'and dist == 1:
                            



                            label = f'{names[int(cls)]}'
                            xyxy_ = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]) - int(xyxy[0]),
                                    int(xyxy[3]) - int(xyxy[1])]
                            # x1, y1 = (int(xyxy[0]) + int(xyxy[2])) / 2, (int(xyxy[1]) + int(xyxy[3])) / 2  # 중심점 좌표 x, y
                            #  크기? 중심점 위치? 를 사용하여 디텍팅 되는 차량을 한정지음

                            x, y, w, h = xyxy_[0], xyxy_[1], xyxy_[2], xyxy_[3]
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                            cv2.putText(im0, str(conf), (xyxy_[0], xyxy_[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

                            fx = x + (w / 2)
                            fy = y + h  # 차 밑 부분 찍는게 맞음 450
                            p_point = np.array([fx, fy, 1], dtype=int)
                            p_point.T
                            np.transpose(p_point)
                            Cal = h1 @ p_point
                            realC = Cal / Cal[2]
                            a = round(int(realC[0]), 0)  # 중심점 x 좌표
                            b = round(int(realC[1]), 0)  # 밑바닥 y 좌표
                            
                            time_save = time.time()
                            
                            # 첫 번째 detection일 때
                            if cam_num == 77 or cam_num == 36:
                                if is_enter(a, b):
                                    # 이전 frame record 없을 때
                                    if len(time_vector[0]) == 0:
                                        time_vector[0].append([time_save, a, b])
                                    else: # record 존재할 때
                                        near_idx = 0
                                        IoU = False
                                        MIN = 10000000
                                        for car_idx in range(0, len(time_vector)):
                                            if abs(a - time_vector[car_idx][-1][1]) < 153 and abs(b-time_vector[car_idx][-1][2]) < 64:
                                                delta = abs(time_save - time_vector[car_idx][-1][0])
                                                if MIN > delta:
                                                    MIN = delta
                                                    IoU = True
                                                    near_idx = car_idx
                                        if IoU:
                                            time_vector[near_idx].append([time_save, a, b])
                                        else:
                                            time_vector.append([])
                                            time_vector[-1].append([time_save, a, b])
                                    
                                    # 입차할 때만 car_list 생성, 트래킹할 때는 car_list 수정
                                    CCTV_OPEN[row][col] += 1
                                    gen_2nd = next_camera(cam_num)
                                    g2_row, g2_col = get_camera_index(gen_2nd)                                   
                                    CCTV_OPEN[g2_row][g2_col] += 1
                                    gen_3rd = next_camera(gen_2nd)
                                    g3_row, g3_col = get_camera_index(gen_3rd)
                                    CCTV_OPEN[g3_row][g3_col] += 1
                                    car_list.append([[cam_num], [gen_2nd], [gen_3rd]])
                            
                            # 하나의 car list에서 2번째 있는 카메라 번호와 일치한지
                            # 새로운 3세대 추가
                            # 기존 1세대 제거, 2->1, 3->2 세대 변환
                            r_idx = 0
                            for r in range(len(car_list)):
                                for node in car_list[r][1]:
                                    if node == cam_num:
                                        r_idx = r
                                        l = []
                                        for third in car_list[r][2]:
                                            gen_3rd = next_camera(third)
                                            g3_row, g3_col = get_camera_index(gen_3rd)
                                            CCTV_OPEN[g3_row][g3_col] += 1
                                            l.append(gen_3rd)
                                        car_list[r].append(l)
                                        g1_row, g1_col = get_camera_index(car_list[r][0][0])
                                        CCTV_OPEN[g1_row][g1_col] -= 1
                                        del car_list[r][0]

                            # 형제 제거
                            for 
                                


                                        

                            
                            # 3세대 OPEN
                            for node in next_camera(cam_num):
                                idx_row, idx_col = get_camera_index(node)
                                gen_2nd.append(node)
                            for node in gen_2nd:
                                idx_row, idx_col = get_camera_index(node)
                                CCTV_OPEN[idx_row][idx_col] += 1

                            # time vector IoU  
                            # 입차 카메라가 아닌 기준
                            near_idx = 0
                            IoU = False
                            MIN = 10000000
                            for car_idx in range(0, len(time_vector)):
                                if abs(a - time_vector[car_idx][-1][1]) < 153 and abs(b-time_vector[car_idx][-1][2]) < 64:
                                    delta = abs(time_save - time_vector[car_idx][-1][0])
                                    if MIN > delta:
                                        MIN = delta
                                        IoU = True
                                        near_idx = car_idx
                            if IoU:
                                time_vector[near_idx].append([time_save, a, b])
                            
                            
                            
                            
                            
                            

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

        t4 = time_synchronized()

        print(f'{s}Done2. ({t4 - t3:.3f}s)')

        t_end = time_synchronized()
        cv2.imshow('frame', im0)
        if cv2.waitKey(27) == ord('w'):
                exit()

#cctv 주소
#'rtsp://admin:admin1234@218.153.209.100:500/cam/realmonitor?channel=1&subtype=1'    500부터 506
#500 : channel = 1 - 30  : 249 11번 - 40번
#501 : channel = 1 - 30  : 250 41번 - 70번
#502 : channel = 1 - 30  : 251 71번 - 100번
#503 : channel = 1 - 30  : 252 101번 - 130번
#504 : channel = 1 - 30  : 253 #순서가 다름 : 
# 132 136 135 139 137 140 138 134 131 133 142 150 143 144 148 145 146 149 147 141 157 159 152 160 156 155 154 151 158 153 
#  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30
#505 : channel = 1 - 30  : 254
#506 : channel = 1 - 30  : 255

PortNumber = [500, 501, 502, 503, 504, 505, 506]
qe    = [10, 40, 70, 100, 130, 160, 190]

CCTV_exist = [[cam for cam in range(11, 41)], #500

             [cam for cam in range(41, 71)], #501

             [cam for cam in range(71, 101)], #502

             [cam for cam in range(101, 131)], #503

            #   [131, 132, 133, 134, 135, 136, 137, 138, 140,
            #   141, 142, 143, 144, 145, 146, 147, 148, 150,
            #   151, 152, 153, 154, 155, 156, 157, 159, 160], #504

             [132, 136, 135, 139, 137, 140, 138, 134, 131, 133,
              142, 150, 143, 144, 148, 145, 146, 149, 147, 141, 
              157, 159, 152, 160, 156, 155, 154, 151, 158, 153], #504

             [cam for cam in range(161, 191)], #505

             [cam for cam in range(191, 221)]] #506] #506
CCTV_OPEN = [[0 for _ in range(30)] for _ in range(len(CCTV_exist))]
TRIGGER = [[0 for _ in range(30)] for _ in range(len(CCTV_exist))]

# optional parameter
parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--project', default='runs/detect', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--nosave', action='store_true', help='do not save images/videos')

opt = parser.parse_args()
print(opt)
check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

weights, imgsz = opt.weights, opt.img_size
view_img = opt.view_img
save_txt = opt.save_txt
# Directories
save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
# Initialize
set_logging()
device = select_device(opt.device)
half = device.type != 'cpu'  # half precision only supported on CUDA
# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size
names = model.module.names if hasattr(model, 'module') else model.names  # get class names
if half:
    model.half()  # to FP16
# Second-stage classifier
classify = False
if classify:
    modelc = load_classifier(name='resnet101', n=2)  # initialize
    modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
# Run inference
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once


# 모든 cctv 채널 OPEN
cam_list = []
CCTV_loading_Start = time_synchronized()
for i, port in enumerate(PortNumber):
    cam_list1 = []
    for j, cam in enumerate(CCTV_exist[i]):
        t1 = time_synchronized()
        source = f'rtsp://admin:admin1234@218.153.209.100:{port}/cam/realmonitor?channel=f{cam-qe[i]}&subtype=1'
        im0s = cv2.VideoCapture(source)
        cam_list1.append(im0s)
        t2 = time_synchronized()
        print(port, cam, "VideoCapture time : ",  f'({t2 - t1:.3f}s)')
    cam_list.append(cam_list1)
CCTV_loading_End = time_synchronized()
print("loading frame time : ",  f'({CCTV_loading_End - CCTV_loading_Start:.3f}s)')

# thread initialize
thread = CCTV_OPEN

# 75번, 39번 카메라만 항상 OPEN
CCTV_OPEN[2][4] = 1
CCTV_OPEN[0][28] = 1
thread[2][4] = Thread(target=watch_cctv, args=(CCTV_exist[i][j], i, j, CCTV_OPEN, CCTV_exist, TRIGGER))
thread[0][28] = Thread(target=watch_cctv, args=(CCTV_exist[i][j], i, j, CCTV_OPEN, CCTV_exist, TRIGGER))

# car list
car_list = []

# 카메라 OPEN -> read
while True:
    for i, port in enumerate(CCTV_OPEN):
        for j, cam in enumerate(CCTV_OPEN[i]):
            if CCTV_OPEN[i][j] > 0 and thread[i][j] == 0: # 이미 쓰레드 동작중이면 다시 실행X
                thread[i][j] = Thread(target=watch_cctv, args=(CCTV_exist[i][j], i, j, CCTV_OPEN, CCTV_exist, car_list))
                thread[i][j].start()
                



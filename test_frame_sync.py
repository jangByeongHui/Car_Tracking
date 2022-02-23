import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import True_, random
import homo_point, parking_position, camera_link
import os, sys, math

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

def numbering_cam(mp4):
    mp4_split = mp4.split('_')
    ip = mp4_split[0][10:]
    channel = int(mp4_split[1][2:])
    if ip == '249':
        channel = channel + 10
    elif ip == '250':
        channel = channel + 40
    elif ip == '251':
        channel = channel + 70

    return channel

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
        x2 = homo_point.entrance[cam_num][0][0]
        y2 = homo_point.entrance[cam_num][0][1]

        if x1 <= x < x2 and y1 < y <= y2:
            return True
    return False

def get_start_time(source, frame):
    file_list = os.listdir(source)
    file_list_mp4 = [file for file in file_list if file.startswith('192.168.8.')]

    mp4_split = file_list_mp4[0].split('_')
    start = mp4_split[2]
    hour = int(start[8:10])
    min = int(start[10:12])
    sec = int(start[12:])
    # finish = mp4_split[4]
    sec += frame*0.05
    if sec >= 60:
        sec = sec - 60
        min = min + 1
        if min >= 60:
            min = min - 60
            hour = hour + 1

    start_time = f'{hour:02d}{min:02d}{sec}'
    # print('###################')
    # print('TIME: ', start_time)
    # print()

    return start_time


parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
parser.add_argument('--source', type=str, default=f'./data/videos/testcase5-2/', help='source')  # file/folder, 0 for webcam
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--project', default='runs/detect', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')

opt = parser.parse_args()
print(opt)
check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    ('rtsp://', 'rtmp://', 'http://', 'https://'))

if webcam:
    print('webcam')
else:
    cam = []
    camera_video_files = []
    contours = []
    contours_ent_list = []
    h1_list = []
    record = []
    tracking_point = {}
    start_time = None
    start_check = 1
    file_list = os.listdir(source)
    file_list_mp4 = [file for file in file_list if file.startswith('192.168.8.')]
    for i, file in enumerate(file_list_mp4):
        file_path = source + file
        camera_video_files.append(file_path)
        cam_num = numbering_cam(file)
        cam.append(cam_num)

        rect_list = np.array(homo_point.pts_dst[cam_num])

        # Create a black image
        img_b = np.zeros((480, 720, 3), dtype=np.uint8)
        img_b = cv2.fillPoly(img_b, [rect_list], (255, 255, 255))
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        res, thr = cv2.threshold(img_b, 127, 255, cv2.THRESH_BINARY)
        contours_b, his = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours.append(contours_b)

        # create a balck image for detecting entrance
        # cam_num = 77
        if cam_num in homo_point.ent_rect:
            ent_list = np.array(homo_point.ent_rect[cam_num])
            img_ent = np.zeros((480, 720, 3), dtype=np.uint8)
            img_ent = cv2.fillPoly(img_ent, [ent_list], (255, 255, 255))
            img_ent = cv2.cvtColor(img_ent, cv2.COLOR_BGR2GRAY)
            res, thr_ent = cv2.threshold(img_ent, 127, 255, cv2.THRESH_BINARY)
            contours_ent, his = cv2.findContours(thr_ent, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Homograpy 처리 후 매핑작업
        im_src = cv2.imread('ch_b1.png')  # 1층
        # im_src = cv2.imread('ch_b2.png')  # 2층

        pts_src = np.array(homo_point.pts_src[cam_num])
        cv2.polylines(im_src, [pts_src], True, (0, 0, 0), 2)
        im_dst = cv2.imread('cam80.png')

        pts_dst = np.array(homo_point.pts_dst[cam_num])
        cv2.polylines(im_dst, [pts_dst], True, (255, 255, 255), 2)
        h1, status = cv2.findHomography(pts_dst, pts_src)
        h1_list.append(h1)

camera_video_files.sort()

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

# Set Dataloader
vid_path, vid_writer = None, None
if webcam:
    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# Run inference
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
t0 = time.time()

# 이미지 처리

# variable 
caps = []
before_frame = [[] for _ in range(len(camera_video_files))]

cnt = [0] * len(camera_video_files)
car_color = [random.randint(0, 255) for _ in range(3)]
for f in range(1000):
    now_frame = [[] for _ in range(len(camera_video_files))]
    for num in range(len(camera_video_files)):
        caps.append(cv2.VideoCapture(camera_video_files[num]))
        if not caps[num].isOpened():
            print("Error opening video. MAIN")
            sys.exit()
        ret, im0s = caps[num].read()
        
        # if f % 10 != 0:
        #     continue
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
        
        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        data_list = []
        person_list = []
        im0 = im0s.copy()
        dst2 = im_src.copy()
        # num detections
        for i, det in enumerate(pred):  # detections per image
            t3 = time_synchronized()
            if webcam:  # batch_size >= 1
                s, im0, frame = f'{i}: ', im0s[i].copy(), caps[num].get(cv2.CAP_PROP_POS_FRAME_COUNT)
            else:
                s, im0, frame = '', im0s.copy(), caps[num].get(cv2.CAP_PROP_POS_FRAMES)

            #  이미지 리사이즈
            im0 = cv2.resize(im0, (720, 480))

            
            s += '%gx%g ' % img.shape[2:]  # print string(img shape)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            # dat = []

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
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        # with open(txt_path + '.txt', 'a') as f:
                        #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or opt.save_crop or view_img:  # Add bbox to image
                        x1, y1 = (int(xyxy[0]) + int(xyxy[2])) / 2, int(xyxy[3])  # 중심점 좌표 x, y
                        
                        
                        dist = cv2.pointPolygonTest(contours[num][0], (x1, y1), False)
                        if cam[num] in homo_point.ent_rect:
                            ent = cv2.pointPolygonTest(contours_ent[0], (x1, y1), False)
                        # if names[int(cls)] == 'car' or names[int(cls)] == 'person':  # box 가 차량이나 사람일 경우
                        if names[int(cls)] == 'car'and dist == 1:
                            if start_check == 1 and ent == 1:
                                # 입차 구역에 들어오고 첫 detection
                                print('start')
                                start_time = get_start_time(source, f)
                                car_color = [random.randint(0, 255) for _ in range(3)]
                                print(start_time)
                                start_check = 0
                            elif start_check == 0 and ent == 1:
                                # 입차 구역에 있는데 이미 start_time이 정해져있을때
                                start_check = 0
                            else:
                                # 입차 구역에서 차가 벗어나고 새로운 차량의 입차 가능성을 열어둠
                                start_check = 1
                            label = f'{names[int(cls)]}'
                            xyxy_ = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]) - int(xyxy[0]),
                                    int(xyxy[3]) - int(xyxy[1])]
                            # x1, y1 = (int(xyxy[0]) + int(xyxy[2])) / 2, (int(xyxy[1]) + int(xyxy[3])) / 2  # 중심점 좌표 x, y
                            #  크기? 중심점 위치? 를 사용하여 디텍팅 되는 차량을 한정지음

                            x, y, w, h = xyxy_[0], xyxy_[1], xyxy_[2], xyxy_[3]
                            # far_persons = cv2.rectangle(im0, (x, y), (x + w, y + h), color, 1)
                            # far_persons_list = np.append(far_persons_list, np.array([[x, y, x + w, y + h]]), axis=0)
                            # print('far_persons_list : %s' % far_persons_list)

                            fx = x + (w / 2)
                            fy = y + h  # 차 밑 부분 찍는게 맞음 450
                            p_point = np.array([fx, fy, 1], dtype=int)
                            p_point.T
                            np.transpose(p_point)
                            Cal = h1_list[num] @ p_point
                            realC = Cal / Cal[2]
                            a = round(int(realC[0]), 0)  # 중심점 x 좌표
                            b = round(int(realC[1]), 0)  # 중심점 y 좌표
                            
                            car_width = 40  # 지도 차량 크기 변수 (2.5m : 64px)
                            car_length = 90 # 6m

                            # cv2.rectangle(dst2, (a - car_length, b - car_width//2), (a , b + car_width//2), (0, 0, -255), car_width)
                            cv2.circle(dst2, (a, b), 20, car_color, -1)
                            cv2.circle(im_src, (a, b), 10, (0, 0, 255), -1)
                            
                            now_frame[num].append([a, b, start_time])
                            start_time = None

                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                            cv2.putText(im0, str(conf), (xyxy_[0], xyxy_[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

            # data_list.append(dat)  # dat: 검출된 box xy 좌표 배열 목록

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
        
        print(f'before_frame{num}: ', before_frame[num])
        print(f'now_frame{num}: ', now_frame[num])

        # 이전 프레임의 ID를 현재 프레임으로 넘겨주기
        idx = 0
        if len(now_frame[num]) <= len(before_frame[num]):
            for i in range(len(now_frame[num])): # 현재 프레임의 차량 검출 개수
                MIN_dist = 10000000 # dummy value
                for j in range(len(before_frame[num])): # 이전 프레임의 차량 검출 개수
                    dist = math.sqrt((before_frame[num][j][0]-now_frame[num][i][0])**2 + (before_frame[num][j][1]-now_frame[num][i][1])**2)
                    if MIN_dist > dist:
                        MIN_dist = dist
                        idx = j
                if len(before_frame[num]) != 0:
                    now_frame[num][i][2] = before_frame[num][idx][2]

                # track point 저장 dictionary 형태
                l = []
                if now_frame[num][i][2] in tracking_point:
                    l = tracking_point[now_frame[num][i][2]]
                l.append(now_frame[num][i])
                tracking_point[now_frame[num][i][2]] = l
        else:
            for i in range(len(before_frame[num])): # 이전 프레임의 차량 검출 개수 1개
                MIN_dist = 10000000 # dummy value
                for j in range(len(now_frame[num])): # 현재 프레임의 차량 검출 개수 2개
                    dist = math.sqrt((before_frame[num][i][0]-now_frame[num][j][0])**2 + (before_frame[num][i][1]-now_frame[num][j][1])**2)
                    if MIN_dist > dist:
                        MIN_dist = dist
                        idx = j
                if len(before_frame[num]) != 0:
                    now_frame[num][idx][2] = before_frame[num][i][2]

            for i in range(len(now_frame[num])):
                # track point 저장 dictionary 형태
                l = []
                if now_frame[num][i][2] in tracking_point:
                    l = tracking_point[now_frame[num][i][2]]
                l.append(now_frame[num][i])
                tracking_point[now_frame[num][i][2]] = l

        for key in tracking_point:
            if len(tracking_point[key]):
                print(f'tracking point[{key}]: ', tracking_point[key])

        # 카메라 넘어가는 시점에 ID 넘겨주기
        next_camera = []
        for key in camera_link.link:
            if key == cam[num]:
                next_camera = camera_link.link[key]
        print(cam[num])
        print('next cam :', next_camera)
        for next in next_camera:
            if next not in cam:
                continue
            if len(now_frame[num]) != 0 and len(before_frame[cam.index(next)]) != 0:
                for index in range(len(before_frame[cam.index(next)])):
                    if before_frame[cam.index(next)][index][0] - now_frame[num][-1][0] < 200 and before_frame[cam.index(next)][index][1] - now_frame[num][-1][1] < 80:
                        before_frame[cam.index(next)][index][2] = now_frame[num][-1][2]
                
    
        # print(f'before_frame{num}: ', before_frame[num])
        # print(f'now_frame{num}: ', now_frame[num])

        if len(now_frame[num]) != 0:
            before_frame[num] = now_frame[num]
            dst2 = cv2.resize(dst2, dsize=(0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
            cv2.imshow("Source Image", dst2)
            cnt[num] = 0
        else:
            # 차량을 잡다가 5번 연속 차량을 못잡을 경우 카메라가 넘어간 것으로 판단
            cnt[num] += 1
            if cnt[num] > 5:
                before_frame[num]= []
                cnt[num] = 0
        #  바둑판 그리기
        # for k in range(0, 14):
        #     cv2.line(im0, (k * 50, 0), (k * 50, 480), (120, 120, 120), 1, 1)
        # for k in range(0, 9):
        #     cv2.line(im0, (0, k * 50), (720, k * 50), (120, 120, 120), 1, 1)
        t4 = time_synchronized()

        print(f'{s}Done2. ({t4 - t3:.3f}s)')

        cv2.imshow('frame', im0)
        # cv2.imshow('maps', dst2)
        t_end = time_synchronized()

        if cv2.waitKey(27) == ord('w'):
            exit()
cv2.imshow("parking", im_src)

  



    
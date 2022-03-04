from multiprocessing import Manager, Process, Queue
import argparse
from platform import java_ver
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import random
import math
import homo_point, parking_position, camera_link
import os

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

def detect(opt, sync, cam_num, record):  # Homography 매칭에 사용되는 행렬값 입력받아야함
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # 호모그래피 destination 이미지 좌표 값
    rect_list = np.array(homo_point.pts_dst[cam_num],np.int32)

    # Create a black image
    img_b = np.zeros((480, 720, 3), dtype=np.uint8)
    img_b = cv2.fillPoly(img_b, [rect_list], (255, 255, 255))
    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    res, thr = cv2.threshold(img_b, 127, 255, cv2.THRESH_BINARY)
    contours_b, his = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Homograpy 처리 후 매핑작업
    im_src = cv2.imread('ch_b1.png')  # 1층
    #호모그래피 Source 이미지 좌표 값
    pts_src = np.array(homo_point.pts_src[cam_num])
    # 표시한 좌표로 연결된 Boundary 표시
    cv2.polylines(im_src, [pts_src], True, (0, 0, 0), 2)

    # 실제 주차장 이미지
    im_dst = cv2.imread('cam80.png')
   #호모그래피 Destination 이미지 좌표 값
    pts_dst = np.array(homo_point.pts_dst[cam_num])
    # 표시한 좌표로 연결된 Boundary 표시
    cv2.polylines(im_dst, [pts_dst], True, (255, 255, 255), 2)
    # Source 좌표와 Destination 좌표를 통한 Homography 변환 행렬 계산
    h1, status = cv2.findHomography(pts_dst, pts_src)

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
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    # data_list = []

    # cctv마다 frame sync가 맞지 않음 20FPS or 30FPS
    frame_sync = -1
    # 차량이 주차장에 들어와 처음 detection된 시점을 구하기 위해
    time_vector = [[]]
    car_length = 153
    car_width = 64

    for path, img, im0s, vid_cap in dataset:  # 이미지 처리 시작, 영상의 경우 한장씩
        # print(dataset[0])
        # frame synchronize
        # frame_sync += 1
        # if frame_sync % sync != 0:
        #     continue
        t0 = time_synchronized()
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
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            t3 = time_synchronized()
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            #  이미지 리사이즈
            im0 = cv2.resize(im0, (720, 480))

            p = Path(p)  # to Path-
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
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
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or opt.save_crop or view_img:  # Add bbox to image
                        x1, y1 = (int(xyxy[0]) + int(xyxy[2])) / 2, int(xyxy[3])  # 중심점 좌표 x, y
                        # pointPolygonTest:(x1,y1) 좌표가 통로 내부에 있다고(등치선) 판단되면은 1, 일치하면 0, 밖에 있으면 -1
                        dist = cv2.pointPolygonTest(contours_b[0], (x1, y1), False)

                        #  객체가 차량(사람)이고 통로 내부에 위치하고 있다면 Homography 연산 실행
                        # if names[int(cls)] == 'car' or names[int(cls)] == 'person' and dist==1:  # box 가 차량이나 사람일 경우
                        if names[int(cls)] == 'car'and dist == 1:
                            label = f'{names[int(cls)]}'
                            xyxy_ = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]) - int(xyxy[0]),
                                    int(xyxy[3]) - int(xyxy[1])]

                            x, y, w, h = xyxy_[0], xyxy_[1], xyxy_[2], xyxy_[3]
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                            cv2.putText(im0, str(f'X:{xyxy_[0]} Y:{xyxy_[1]}'), (xyxy_[0]+(w/2), xyxy_[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)

                            fx = x + (w / 2)
                            fy = y + h  # 차 밑 부분 찍는게 맞음 450

                            # Homography를 통한 좌표 변환
                            p_point = np.array([fx, fy, 1], dtype=int)
                            # p_point.T
                            np.transpose(p_point)
                            Cal = h1 @ p_point
                            realC = Cal / Cal[2]
                            a = round(int(realC[0]), 0)  # 중심점 x 좌표
                            b = round(int(realC[1]), 0)  # 밑바닥 y 좌표

                            # 이전 frame record 없을 때
                            if len(time_vector[0]) == 0:
                                time_vector[0].append([frame_sync, a, b])
                            else: # record 존재할 때
                                near_idx = 0
                                IoU = False
                                MIN = 10000000
                                for car_idx in range(0, len(time_vector)):
                                    if abs(a - time_vector[car_idx][-1][1]) < 153 and abs(b-time_vector[car_idx][-1][2]) < 64:
                                        delta = abs(frame_sync - time_vector[car_idx][-1][0])
                                        if MIN > delta:
                                            MIN = delta
                                            IoU = True
                                            near_idx = car_idx
                                if IoU:
                                    time_vector[near_idx].append([frame_sync, a, b])
                                else:
                                    time_vector.append([])
                                    time_vector[-1].append([frame_sync, a, b])

            # data_list.append(dat)  # dat: 검출된 box xy 좌표 배열 목록

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

        t4 = time_synchronized()

        print(f'{s}Done2. ({t4 - t3:.3f}s)')

        t_end = time_synchronized()
        print(f'{frame_sync} of total time: ', t_end-t0)
        cv2.imshow('frame', im0)
        if cv2.waitKey(27) == ord('w'):
                exit()

    record.append(time_vector)

def execute_tracking(cam_num, frame_sync, record):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    if 11<= cam_num <= 40:
        cam = cam_num - 10
        channel = f'192.168.8.249_ch{cam}_'
    elif 41 <= cam_num <= 70:
        cam = cam_num - 40
        channel = f'192.168.8.250_ch{cam}_'
    elif 71<= cam_num <= 100:
        cam = cam_num - 70
        channel = f'192.168.8.251_ch{cam}_'
    
    # start = '20210722093015' # case2
    # start = '20210801120905' # case 3
    # start = '20210728092930' # case 4
    # start = '20210728082210' # case 5-1
    # start = '20210728095145' # case 5-2
    start = '20210728134600' # caset 5-3
    # finish = '20210722093100' # case2
    # finish = '20210801120955' # case 3
    # finish = '20210728093040' # case 4
    # finish = '20210728082232' # case 5-1
    # finish = '20210728095245' # case 5-2
    finish = '20210728134635' # case 5-3
    file = channel + start + '_' + finish
    # 192.168.8.251_ch7_20210801120905_20210801120955

    # parser.add_argument('--source', type=str, default=f'rtsp://admin:admin1234@218.153.209.100:502/cam/realmonitor?channel={cam}&subtype=1', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default=f'./data/videos/testcase5-3/{file}.mp4', help='source')  # file/folder, 0 for webcam
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
    # parser.add_argument('--totalmap-conf', default='[[7736, 1688], [7396, 1686], [7402, 1439], [7720, 1442]]',
    #                     help='totalmap')
    # parser.add_argument('--map-conf', default='[[202, 182], [259, 117], [448, 118], [506, 183]]',
    #                     help='totalmap')

    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(opt=opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt=opt, sync=frame_sync, cam_num=cam_num, record=record)

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

def get_start_time(start, time):
    year = int(start[0:4])
    month = int(start[4:6])
    day = int(start[6:8])
    h = int(start[8:10])
    m = int(start[10:12])
    s = int(start[12:])

    s = s + time
    if s >= 60:
        m += 1
        s -= 60
        if m >= 60:
            h += 1
            m -= 60
    
    result = f'{year}{month:02d}{day:02d} {h:02d}:{m:02d}:{s:04.1f}'
    return result


def next_camera(cam):
    next_camera = []
    for key in camera_link.link:
        if key == cam:
            next_camera = camera_link.link[key]
            break
    
    return next_camera

def check_iou(x1, y1, x2, y2):
    l = 153 # car length
    d = 64 # car width
    if abs(x2-x1) < l and abs(y2-y1) < d:
        return True
    else:
        return False

if __name__ == '__main__':

    # 프로세스를 생성
    p = []
    record = []
    id = Manager()

    # Camera MP4
    test_camera = [77, 78, 79, 80, 24] # test case 5-1, 5-3 ->[77,78,79,80,24], test case 5-2 ->[77,78,79,80]
    # 입차 시간
    # start = '20210722093015' # case2
    # start = '20210801120905' # case 3
    # start = '20210728092930' # case 4
    # start = '20210728082210' # case 5-1
    # start = '20210728095145' # case 5-2
    start = '20210728134600' # caset 5-3

    sync = 2 # 원본 영상 20FPS
    for idx in range(len(test_camera)):
        record.append(Manager().list())
        # sync = 1 # 20FPS 
        p.append(Process(target=execute_tracking, args=(test_camera[idx], sync, record[idx])))

    # process 시작
    for idx in range(len(test_camera)):
        p[idx].start()

    # process가 작업을 완료할 때까지 기다림
    for idx in range(len(test_camera)):
        p[idx].join()

    for i in range(len(test_camera)):
        print(f'RECORD{test_camera[i]}: ', record[i])


    track_point = []
    temp = record[0][0] # 5 -> 3 dim
    
    for i in range(len(temp)):
        track_point.append([])
        track_point[i] = temp[i]

    # 카메라 넘어가는 부분 array 스티칭
    for n in range(1, len(test_camera)):
        temp = record[n][0]
        for i in range(len(track_point)): 
            MIN = 10000000
            idx = -1
            for j in range(len(temp)):
                frame_gap = abs(track_point[i][-1][0] - temp[j][0][0])
                if MIN > frame_gap:
                    MIN = frame_gap
                    idx = j
            if track_point[i][-1][0] <= temp[idx][0][0]:
                if check_iou(track_point[i][-1][1], track_point[i][-1][2], temp[idx][0][1], temp[idx][0][2]):
                    track_point[i].extend(temp[idx])
            else:
                s = len(track_point[i])-1
                f = s - (track_point[i][-1][0] - temp[idx][0][0])//sync
                if f < 0: f = 0
                for k in range(s, f, -1): # 20FPS: //1, 10FPS: //2
                    
                    if check_iou(track_point[i][k][1], track_point[i][k][2], temp[idx][0][1], temp[idx][0][2]):
                        
                        track_point[i].extend(temp[idx])
                        break

    im_src = cv2.imread('ch_b1.png')
    car_color = [[0,0,255],[255,0,0],[0,255,0],[255,255,0],[0,255,255]]

    

    # 입치시간, 주차 위치, 주차시 카메라 번호 확인
    # 주차장 MAP에 차량 이동 경로 추적 
    for i in range(len(track_point)):
        
        if is_enter(track_point[i][0][1], track_point[i][0][2]):
            time = 0.05 * track_point[i][0][0] # 20fps: 0.05 sec per frame
            InParkingLot = get_start_time(start, time)
            print('###################')
            print(f'[{i}]car InPARKING time: ', InParkingLot)
            print('###################')

        print(f'track_point[{i}]', track_point[i])
        for j in range(len(track_point[i])):
            cv2.circle(im_src, (track_point[i][j][1], track_point[i][j][2]), 10, car_color[i], -1)
        x = track_point[i][-1][1]
        y = track_point[i][-1][2]
        pos = find_parking_position(x, y)
        if pos:
            cam_num = find_parking_cam_num(x, y)
            cv2.putText(im_src,f'Position: {pos}, CAM_NUM: {cam_num}', (x, y), cv2.FONT_HERSHEY_PLAIN, 10, car_color[i], 2, cv2.LINE_AA)

    dst2 = cv2.resize(im_src, dsize=(0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
    cv2.imshow('parking lot', dst2)
    
    cv2.waitKey(20000)
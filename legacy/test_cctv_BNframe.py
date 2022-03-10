from multiprocessing import Manager, Process, Queue
import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import random
import math
import homo_point, api

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

# global variable 

def detect(opt, queue, sync, cam_num, record):  # Homography 매칭에 사용되는 행렬값 입력받아야함
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size,opt
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Homography 변환을 위한 매칭점 load
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
    
    start_time = ''
    source_split = source.split('_') # source = 192.168.8.251_ch{cam}_20210722093015_20200722093100
    year = source_split[2][:4]
    month = source_split[2][4:6]
    day = source_split[2][6:8]
    hour = int(source_split[2][8:10])
    min = int(source_split[2][10:12])
    sec = int(source_split[2][12:])

    for path, img, im0s, vid_cap in dataset:  # 이미지 처리 시작, 영상의 경우 한장씩
        
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
        frame_sync += 1
        if frame_sync % sync != 0:
            continue
        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        data_list = []
        person_list = []
        im0 = im0s.copy()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
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
                        # rect_list = np.array([[80, 411], [283, 113], [435, 115], [620, 410]], np.int32)  # 차량 검출 범위 박스
                        # # Create a black image
                        # img_b = np.zeros((480, 720, 3), dtype=np.uint8)
                        # img_b = cv2.polylines(img_b, [rect_list], True, (255, 255, 255), 4)
                        # cv2.imshow('testsss', img_b)
                        # cv2.waitKey(0)

                        # img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
                        # res, thr = cv2.threshold(img_b, 127, 255, cv2.THRESH_BINARY)
                        # contours_b, his = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                        # cv2.drawContours(img_b, contours_b[0], -1, (0, 255, 0), 4)
                        # cv2.imshow('testsss', img_b)
                        # cv2.waitKey(0)
                        
                        dist = cv2.pointPolygonTest(contours_b[0], (x1, y1), False)

                        # if names[int(cls)] == 'car' or names[int(cls)] == 'person':  # box 가 차량이나 사람일 경우
                        if names[int(cls)] == 'car'and dist == 1:
                            # 1번째 detection만 동작
                            if not start_time and cam_num == 77:
                                sec += frame_sync*0.05
                                if sec >= 60:
                                    sec = sec - 60
                                    min = min + 1
                                    if min >= 60:
                                        min = min - 60
                                        hour = hour + 1
                                start_time = f'{hour:02d}{min:02d}{sec}'
                                print('###################')
                                print('TIME: ', start_time)
                                print()
                                record.append(start_time)
                                
                            label = f'{names[int(cls)]}'
                            xyxy_ = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]) - int(xyxy[0]),
                                    int(xyxy[3]) - int(xyxy[1])]
                            data_list.append(xyxy_)
                            # x1, y1 = (int(xyxy[0]) + int(xyxy[2])) / 2, (int(xyxy[1]) + int(xyxy[3])) / 2  # 중심점 좌표 x, y
                            #  크기? 중심점 위치? 를 사용하여 디텍팅 되는 차량을 한정지음

                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                            cv2.putText(im0, str(conf), (xyxy_[0], xyxy_[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

            # data_list.append(dat)  # dat: 검출된 box xy 좌표 배열 목록

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
        t3 = time_synchronized()

        # cv2.imshow('maps', im_src)

        far_persons_list = np.empty((0, 4), int)
        near_persons_list = np.empty((0, 4), int)

        if len(data_list) != 0:
            dst2 = im_src.copy()

        for i in range(len(data_list)):  # boxes: yolo 검출된 box 의 x, y, w, h
            x, y, w, h = data_list[i][0], data_list[i][1], data_list[i][2], data_list[i][3]
            color = (0, 0, 255)

            # far_persons = cv2.rectangle(im0, (x, y), (x + w, y + h), color, 1)
            far_persons_list = np.append(far_persons_list, np.array([[x, y, x + w, y + h]]), axis=0)
            # print('far_persons_list : %s' % far_persons_list)

            fx = x + (w / 2)
            fy = y + h  # 차 밑 부분 찍는게 맞음 450
            p_point = np.array([fx, fy, 1], dtype=int)
            p_point.T
            np.transpose(p_point)
            Cal = h1 @ p_point
            # global ArealC
            # ArealC = Cal / Cal[2]
            # a = round(int(ArealC[0]), 0)  # 중심점 x 좌표
            # b = round(int(ArealC[1]), 0)  # 중심점 y 좌표
            realC = Cal / Cal[2]
            a = round(int(realC[0]), 0)  # 중심점 x 좌표
            b = round(int(realC[1]), 0)  # 중심점 y 좌표
            
            queue.put([a, b])
            car_width = 40  # 지도 차량 크기 변수 (2.5m : 64px)
            car_length = 90 # 6m

            cv2.rectangle(dst2, (a - car_length, b - car_width//2), (a , b + car_width//2), (0, 0, -255), car_width)
            # cv2.circle(im_src, (a, b), 10, (0, 0, -255), -1)

        if len(data_list) != 0:
            dst2 = cv2.resize(dst2, dsize=(0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
            # cv2.imshow("Source Image", dst2)
        #  바둑판 그리기
        # for k in range(0, 14):
        #     cv2.line(im0, (k * 50, 0), (k * 50, 480), (120, 120, 120), 1, 1)
        # for k in range(0, 9):
        #     cv2.line(im0, (0, k * 50), (720, k * 50), (120, 120, 120), 1, 1)
        t4 = time_synchronized()

        print(f'{s}Done2. ({t4 - t3:.3f}s)')

        cv2.imshow('frame', im0)
        
        if cv2.waitKey(27) == ord('w'):
            exit()

    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     print(f"Results saved to {save_dir}{s}")
    #
    # print(f'Done. ({time.time() - t0:.3f}s)')
def execute_tracking(q_num, cam_num, frame_sync, record):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    cam = cam_num % 70
    channel = f'192.168.8.251_ch{cam}_'
    start = '20210722093015'
    finish = '20210722093100'
    file = channel + start + '_' + finish
    # parser.add_argument('--source', type=str, default=f'rtsp://admin:admin1234@218.153.209.100:502/cam/realmonitor?channel={cam}&subtype=1', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default=f'./data/videos/{file}.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--homo-ch', type=int, default=79, help='보고자하는 CCTV 구역 숫자 입력 ex) 75,77,80...')
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
            detect(opt=opt, queue=q_num, sync=frame_sync, cam_num=cam_num, record=record)

def find_parking_position(x, y):
    spot = api.get_parking_spot() # 한번 call할때마다 api를 불러오는 것이 비효율적이라면 parking_position.py에서 가져다 쓰는 것이 더 좋은 방법
    width = 64
    height = 128
    margin = 3
    position = ''
    for key in spot:
        top = spot[key]['top']
        left = spot[key]['left']
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

def extract_from_queue(q):
    
    now_frame = [[]]
    for process in range(len(q)):
        id = process
        while q[process].qsize() != 0:                
            point = q[process].get()
            a = point[0]
            b = point[1]
            now_frame[process].append([a,b,id])
            id += 1
        now_frame.append([])
    return now_frame

def format_frame(num):
    frame = [[]]
    for i in range(num):
        frame.append([])
    
    return frame

def gui(q, record):
    im_src = cv2.imread('ch_b1.png')
    
    car_width = 40
    car_length = 113
    tracking_point = [[]]
    before_frame = [[]]
    for i in range(20):
        tracking_point.append([])
    for i in range(len(q)):
        before_frame.append([])
    end_flag = 0
    while True:
        dst2 = im_src.copy()
        
        now_frame = extract_from_queue(q)

        # # 차량이 입차인지 아닌지
        # if len(now_frame[0]) != 0 and len(before_frame[0]) != 0: # process 번호는 상황에 맞게 변경시켜줘야함
        #     x = now_frame[0][0][0]
        #     y = now_frame[0][0][1]
        #     if 5736 <= x < 6198 and  1693 <= y < 1821: # 입차 구역 범위
        #         # start_time = get_start_time()
        #         start_time = '09:30:26'
        #         print('###############')
        #         print(start_time)
        #         print('###############')


        # 이전 프레임의 ID를 현재 프레임으로 넘겨주기
        idx = 0
        for process in range(len(q)): # Detection 중인 카메라 개수
            for i in range(len(now_frame[process])): # 현재 프레임의 차량 검출 개수
                MIN_dist = 10000000 # dummy value
                for j in range(len(before_frame[process])): # 이전 프레임의 차량 검출 개수
                    dist = math.sqrt((before_frame[process][j][0]-now_frame[process][i][0])**2 + (before_frame[process][j][1]-now_frame[process][i][1])**2)
                    if MIN_dist > dist:
                        MIN_dist = dist
                        idx = j
                if len(before_frame[process]) != 0:
                    now_frame[process][i][2] = before_frame[process][idx][2]

                # track point 저장
                tracking_point[now_frame[process][i][2]].append(now_frame[process][i])
            
        for id in range(0, 20):
            if len(tracking_point[id]) != 0:
                print(f'tracking point[{id}]: ', tracking_point[id])
        # 카메라 넘어가는 시점에 ID 넘겨주기
        for process in range(1, len(q)):
            if len(now_frame[process]) != 0 and len(before_frame[process-1]) != 0:
                id = before_frame[process-1][-1][2]
                if now_frame[process][0][0] - tracking_point[id][-1][0] < 153:
                    now_frame[process][0][2] = id
                    # id 넘겨줬으면 pop
                    # before_frame[process-1].pop()

        
        # 주차장 지도에 출력
        for process in range(len(q)):

            print(f'before_frame{process}: ', before_frame[process])
            print(f'now_frame{process}: ', now_frame[process])

            if len(now_frame[process]) != 0:
                before_frame[process] = now_frame[process]
            #     for i in range(len(now_frame[process])):
            #         cv2.circle(dst2, (now_frame[process][i][0], now_frame[process][i][1]), 10, (0, 0, 255), -1)
            #         cv2.putText(dst2, f'{now_frame[process][i][2]}', (now_frame[process][i][0], now_frame[process][i][1]), cv2.FONT_HERSHEY_SIMPLEX, 20, (0,0,255), 2)
        #########################################################################################
        for id in range(0, 20):
            for i in range(len(tracking_point[id])):
                
                cv2.circle(dst2, (tracking_point[id][i][0], tracking_point[id][i][1]), 10, (0, 0, 255), -1)
                # cv2.putText(dst2, f'{tracking_point[id][0][2]}', (tracking_point[id][i][0], tracking_point[id][i][1]),cv2.FONT_HERSHEY_SIMPLEX, 20, (0,0,255), 2)
        #########################################################################################
        # trakcing_point의 마지막점을 통해 주차 기둥 위치 탐색
        for id in range(0, 20):
            if len(tracking_point[id]) != 0:
                pos = find_parking_position(tracking_point[id][-1][0], tracking_point[id][-1][1])
                if pos:
                    print('##################')
                    print('parking Lot: ', pos)
                    print('##################')
                    cam_num = find_parking_cam_num(tracking_point[id][-1][0], tracking_point[id][-1][1])
                    print('CAM NUM: ', cam_num)
                    print('##################')
                    cv2.putText(dst2,f'Position: {pos}, CAM_NUM: {cam_num}', (tracking_point[id][-1][0], tracking_point[id][-1][1]), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 2, cv2.LINE_AA)
                    print('start time: ', record[0])
                    
                    record.append(pos)
                    record.append(cam_num)
                    end_flag = 1
                # tracking 차량의 마지막 카메라 위치
                
        #########################################################################################
        dst2 = cv2.resize(dst2, dsize=(0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
        cv2.imshow('Parking lot', dst2)
        cv2.waitKey(1)

        if end_flag == 1:
            time.sleep(10)
            break
if __name__ == '__main__':

    # 프로세스를 생성합니다
    q = []
    p = []

    record = Manager().list()
    test_camera = [77, 78, 79, 80]
    for idx in range(len(test_camera)):
        q.append(Queue())
        sync = 2*2 # 20FPS 
        if test_camera[idx] == 79:
            sync = 3*2 # 30FPS
        p.append(Process(target=execute_tracking, args=(q[idx], test_camera[idx], sync, record)))
    p_gui = Process(target=gui, args=(q, record))

    # process 시작
    for idx in range(len(test_camera)):
        p[idx].start()
    p_gui.start()

    # process가 작업을 완료할 때까지 기다림
    for idx in range(len(test_camera)):
        p[idx].join()
    p_gui.join()

    print('RECORD: ', record)
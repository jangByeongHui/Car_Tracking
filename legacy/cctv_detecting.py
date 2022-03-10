import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import True_, random
import homo_point

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(opt):  # Homography 매칭에 사용되는 행렬값 입력받아야함
    # 각 필요한 파라미터 입력
    source, weights, view_img, save_txt, imgsz,homo_ch = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size,opt.homo_ch

    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    #스트리밍 주소 추출
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    rect_list = np.array(homo_point.pts_dst[homo_ch]) #homo_point.pts_dst[num] 기본 입력

    # Create a black image
    img_b = np.zeros((480, 720, 3), dtype=np.uint8) #720 * 480 사이즈 이미지 추출
    img_b = cv2.fillPoly(img_b, [rect_list], (255, 255, 255)) # 흰색으로 된 다각형 그리기
    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY) # GRAYSCALE로 이미지 변환
    res, thr = cv2.threshold(img_b, 127, 255, cv2.THRESH_BINARY) # BINARY화 특정 임계값에 도달하지 못할시 0으로 바꿈
    contours_b, his = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # 윤곽선 추출

    # Homograpy 처리 후 매핑작업
    im_src = cv2.imread('ch_b1.png')  # 1층 주차장 도면
    pts_src = np.array(homo_point.pts_src[homo_ch]) # 주차장 매칭점
    cv2.polylines(im_src, [pts_src], True, (0, 0, 0), 2) # 주차장 매칭점 Bound 표시
    im_dst = cv2.imread('cam80.png')

    pts_dst = np.array(homo_point.pts_dst[homo_ch]) # 영상 속 주차장 매칭점
    cv2.polylines(im_dst, [pts_dst], True, (255, 255, 255), 2) # 영상 속 매칭점 Bound 표시
    h1, status = cv2.findHomography(pts_dst, pts_src) # Homography 값 계산
    
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

    # 가중치 절반 사용
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

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        data_list = []
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
                        dist = cv2.pointPolygonTest(contours_b[0], (x1, y1), False) # 임의점과 다각형의 거리 계산

                        if names[int(cls)] == 'car' and dist == 1: # 임의로 정해놓은 Bound안에 들어온 차량만 표시
                            label = f'{names[int(cls)]}'
                            xyxy_ = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]) - int(xyxy[0]),
                                     int(xyxy[3]) - int(xyxy[1])]

                            data_list.append(xyxy_) # 조건을 만족하는 차량만 data_list에 추가
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2) # Bounding Box 표시

                            # 차량 탐지 정확도 출력
                            cv2.putText(im0, str(conf), (xyxy_[0], xyxy_[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA) #Conf 표시

            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
        t3 = time_synchronized()

        far_persons_list = np.empty((0, 4), int)
        near_persons_list = np.empty((0, 4), int)

        # 이미지 복사
        if len(data_list) != 0:
            dst2 = im_src.copy()

        # 검출된 객체들의 좌표 Homography 변환
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
            realC = Cal / Cal[2]
            print(realC) # Homography 변환후 계산된 좌표값

            a = round(int(realC[0]), 0)  # 중심점 x 좌표
            b = round(int(realC[1]), 0)  # 중심점 y 좌표

            car_width = 40 # 2.5m 64px 
            car_length = 113 # 6m 153px line width 40
            cv2.rectangle(dst2, (a - car_length, b - car_width//2), (a , b + car_width//2), (0, 0, -255), car_width)
            # cv2.circle(im_src, (a, b), 10, (0, 0, -255), -1)

        if len(data_list) != 0:
            dst2 = cv2.resize(dst2, dsize=(0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
            cv2.imshow("Source Image", dst2)
        t4 = time_synchronized()

        print(f'{s}Done2. ({t4 - t3:.3f}s)')

        cv2.imshow('frame', im0)
        
        if cv2.waitKey(27) == ord('w'):
            exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/videos/79.mp4', help='source')  # file/folder, 0 for webcam
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
            detect(opt=opt)

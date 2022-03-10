import cv2
import numpy as np

oldx = oldy = -1  # 좌표 기본값 설정


def on_mouse(event, x, y, flags, param):
    # event는 마우스 동작 상수값, 클릭, 이동 등등
    # x, y는 내가 띄운 창을 기준으로 좌측 상단점이 0,0이 됌
    # flags는 마우스 이벤트가 발생할 때 키보드 또는 마우스 상태를 의미, Shif+마우스 등 설정가능
    # param은 영상이룻도 있도 전달하고 싶은 데이타, 안쓰더라도 넣어줘야함

    global oldx, oldy
    # 밖에 있는 oldx, oldy 불러옴

    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽이 눌러지면 실행 
        oldx, oldy = x, y
        # 마우스가 눌렀을 때 좌표 저장, 띄워진 영상에서의 좌측 상단 기준

        print('[%d, %d],' % (x, y), end=" ")  # 좌표 출력

        cv2.circle(im_src, (x, y), 3, (255, 0, 0), -1)

    if event == cv2.EVENT_RBUTTONDOWN:  # 오른쪽이 눌러지면 실행
        oldx, oldy = x, y
        # 마우스가 눌렀을 때 좌표 저장, 띄워진 영상에서의 좌측 상단 기준

        print('[%d, %d]' % (x, y), end="],\n")  # 좌표 출력

        cv2.circle(im_src, (x, y), 3, (255, 0, 0), -1)

    if event == cv2.EVENT_MBUTTONDOWN:  # 왼쪽이 눌러지면 실행
        print('\t\t\t  [', end="")


im_src = cv2.imread('cam80.png')
# img2 = cv2.resize(im_src, dsize=(0,0),fx=1/10,fy=1/10, interpolation=cv2.INTER_LINEAR) 
# im_src = cv2.imread('CHN11.PNG')
# pts_src = np.array([[991, 284], [957, 283], [956, 239], [993, 239]])
# cv2.polylines(im_src, [pts_src], True, (0, 0, 0), 2)

cv2.namedWindow('test')
cv2.setMouseCallback('test', on_mouse, im_src)
cv2.imshow('test', im_src)
cv2.waitKey(0)

k = cv2.waitKey(1) or 0xff

if k == 32:
    cv2.waitKey()

if k == 27:
    exit()

# im_src = cv2.imread('abs_test1.png')
# im_src1 = cv2.imread('abs_test2.png')
# im_src = cv2.resize(im_src, (720, 480))
# im_src1 = cv2.resize(im_src1, (720, 480))
#
# absdiff_frame = cv2.absdiff(im_src, im_src1)
# cv2.imshow('abs', absdiff_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np
import glob

img_array = []
img_list = glob.glob('runs/detect/MAP/*.jpg')
img_list.sort()
for filename in img_list:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('MAP.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
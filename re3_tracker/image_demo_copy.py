


import glob
import os
import cv2
from tracker import re3_tracker
tracker = re3_tracker.Re3Tracker()

pathname='D:\Anusha\Pedastrian'
bounding_box=[]
image_paths = sorted(glob.glob(os.path.join(pathname, '*.jpeg')))
#groundtruthpath='D:\Anusha\OTBData\\'+name+'\\groundtruth_rect.txt'
#mydata = pd.read_table(groundtruthpath, sep=',', h eader=None)
name='pedastrian'
box=[0,0,0,0]
box[0]=717  # xmin
box[1]=210  # ymin
box[2]=148  # width
box[3]=111   # height
initial_bbox = [int(box[0]), int(box[1]), int(box[0])+int(box[2]), int(box[1])+int(box[3])]
tracker.track(name, image_paths[0], initial_bbox)
for image_path in image_paths:
   image = cv2.imread(image_path)
   # Tracker expects RGB, but opencv loads BGR.
   imageRGB = image[:, :, ::-1]
   # print("imageRGB:", imageRGB)
   bbox, trackeddata = tracker.track(name, imageRGB)
   # print(">>>", bbox, trackeddata)
   bounding_box.append(bbox)
   cv2.rectangle(image,
                 (int(bbox[0]), int(bbox[1])),
                 (int(bbox[2]), int(bbox[3])),
                 [0, 0, 255], 2)
   cv2.imshow('Image', image)
   cv2.waitKey()





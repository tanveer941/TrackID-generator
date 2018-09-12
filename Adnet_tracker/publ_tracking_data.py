

import ecal
import sys
import AlgoInterface_pb2
import time
import cv2


# @timeit
def publ_signal_names():

    print("Publish tracking data......")

    ecal.initialize(sys.argv, "Tracker data")

    publisher_roi_obj = ecal.publisher(topic_name="SR_Request")
    label_req_obj = AlgoInterface_pb2.LabelRequest()
    # image_path = r'D:\Work\2018\code\Tensorflow_code\Tracker\DLBasedSmartAnnotation\data\BlurCar2\img\0001.jpg'
    image_path = r'D:\Work\2018\code\Tensorflow_code\Tracker\DLBasedSmartAnnotation\data\LT52\img\0001.jpeg'
    # read the image
    # img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_encoded = cv2.imencode('.png', img)[1].tostring()
    # print("img_encoded ::", img)
    img_str = img_encoded
    # Read its respective coordinates
    gt_boxes_path = r'D:\Work\2018\code\Tensorflow_code\Tracker\DLBasedSmartAnnotation\data\BlurCar2\groundtruth_rect.txt'
    image_nmber_idx = 1 - 1
    with open(gt_boxes_path, 'r') as f:
        lines = f.readlines()
    line = lines[image_nmber_idx]
    img_trackid_lst = [2]
    x, y, w, h = [int(x) for x in line.split()]
    # 227	207	122	99
    # print("ordinate:", x, y, w, h)
    # (xmin, ymin, xmax, ymax) = (227, 207, 349, 306)
    # (xmin, ymin, xmax, ymax) = (227, 204, 349, 306)
    (xmin, ymin, xmax, ymax) = (622, 180, 660, 335)
    # xmin = 227
    # ymin = 207
    # xmax = 349
    # ymax = 306
    print("ordinate::", xmin, ymin, xmax, ymax)
    label_req_obj.NextImg.imageData = img_str
    for evy_img_trkid in img_trackid_lst:
        # Assign the trackid here
        attrib_typ_obj = label_req_obj.CurrentAttr.add()
        attrib_typ_obj.trackID = evy_img_trkid
        # Loop in to send across the coordinates
        roi_min_obj = attrib_typ_obj.ROI.add()
        roi_min_obj.X = xmin
        roi_min_obj.Y = ymin
        roi_max_obj = attrib_typ_obj.ROI.add()
        roi_max_obj.X = xmax
        roi_max_obj.Y = ymax

    time.sleep(1)
    publisher_roi_obj.send(label_req_obj.SerializeToString())

#=================================================================================================================

if __name__ == '__main__':

    publ_signal_names()
    # subscribe_signl_vals()
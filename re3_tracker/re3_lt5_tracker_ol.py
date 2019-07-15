"""

pyinstaller.exe --onefile re3_lt5_tracker.py --add-data _ecal_py_3_5_x64.pyd;.
pyinstaller.exe re3_lt5_tracker.py --add-data _ecal_py_3_5_x64.pyd;.
copy re3_topics.json  folder - logs, model
D:\Work\2018\code\Tensorflow_code\Protobuf_compilers\protoc3.5\bin\protoc.exe -I=.\ --python_out=.\ algointerface.proto

"""

import ecal
import algointerface_pb2
import sys
import os
import json
import time
import numpy as np
import cv2
from PIL import Image
from tracker import re3_tracker

TOPICS_JSON = r're3_topics.json'

if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(__file__)
# print("application_path ::", application_path)
topics_json_path = os.path.join(application_path, TOPICS_JSON)

BEXIT = True
ALGO_READINESS = True

PREV_IMAGE = None

class Re3TrackerLT5(object):

    def __init__(self):

        self.tracker = re3_tracker.Re3Tracker()
        self.trkid_lst = []

        # Initialize eCAL
        ecal.initialize(sys.argv, "object tracking")
        # Read the JSON files
        with open(topics_json_path) as data_file:
            self.json_data = json.load(data_file)

        # Topic names
        self.tracker_request = self.json_data['image_request']
        self.tracker_response = self.json_data['image_response']

        print("Tracker initialization done!!")

        # Define the callbacks for publisher subscriber
        self.initialize_subscr_topics()
        self.initialize_publsr_topics()

        # The callbacks will redirect to the tracker function and publish predicted ROI
        self.define_subscr_callbacks()



    def initialize_subscr_topics(self):
        # Initialize all the subscriber topics
        # self.lt5_img_subscr_obj = ecal.subscriber(self.json_data['image_request'])
        self.lt5_img_subscr_obj = ecal.subscriber(self.tracker_request)
        self.lt5_finl_subscr_obj = ecal.subscriber(self.json_data['algo_end_response'])

    def initialize_publsr_topics(self):
        # Initialize all the publisher topics
        # self.lt5_track_publr_obj = ecal.publisher(self.json_data['image_response'])
        self.lt5_track_publr_obj = ecal.publisher(self.tracker_response)
        self.lt5_algo_publr_obj = ecal.publisher(self.json_data['algo_begin_response'])

    def publish_tracked_data(self, multi_track_bbox_dict):

        # print("multi_track_bbox_dict>>", multi_track_bbox_dict)
        if multi_track_bbox_dict:
            lbl_response_obj = algointerface_pb2.LabelResponse()
            for ech_trk_id, bbox_lst in multi_track_bbox_dict.items():
            #==========================================================
                channel_obj = lbl_response_obj.channelobject.add()
                tmstamp_obj = channel_obj.timestampobject.add()
                attrib_typ_obj = tmstamp_obj.CurrentAttr.add()
                attrib_typ_obj.trackID = ech_trk_id
                attrib_typ_obj.hasUserCorrected = 2

                attrib_typ_obj.trackername = 'Re3'
                # bbox_obj in format x,y,w,h
                xmin = bbox_lst[0]
                ymin = bbox_lst[1]
                xmax = bbox_lst[2]
                ymax = bbox_lst[3]
                # print("coordinates::", xmin, ymin, xmax, ymax)
                attrib_typ_obj.height = ymax - ymin
                attrib_typ_obj.width = xmax - xmin
                attrib_typ_obj.hasUserCorrected = 2
                attrib_typ_obj.shape = "Box"
                # Create ROI object for Xmin, Ymin
                roi_min_obj1 = attrib_typ_obj.ROI.add()
                roi_min_obj1.X = xmin
                roi_min_obj1.Y = ymin

                roi_min_obj2 = attrib_typ_obj.ROI.add()
                roi_min_obj2.X = xmax
                roi_min_obj2.Y = ymin

                # Create ROI object for Xmax, Ymax
                roi_max_obj3 = attrib_typ_obj.ROI.add()
                roi_max_obj3.X = xmax
                roi_max_obj3.Y = ymax

                roi_max_obj4 = attrib_typ_obj.ROI.add()
                roi_max_obj4.X = xmin
                roi_max_obj4.Y = ymax

            self.lt5_track_publr_obj.send(lbl_response_obj.SerializeToString())

    def predict_tracker_result(self, topic_name, msg, time):
        global ALGO_READINESS
        ALGO_READINESS = False
        lbl_request_obj = algointerface_pb2.LabelRequest()
        multi_track_bbox_dict = {}
        if msg is not None:
            lbl_request_obj.ParseFromString(msg)
            for ech_chnl_obj in lbl_request_obj.channelobject:
                chnl_name = ech_chnl_obj.channelName
                # print("chnl_name:>", chnl_name)
                for ech_tmstamp_obj in ech_chnl_obj.timestampobject:
                    tm_stamp = ech_tmstamp_obj.timestamp
                    # print("tm_stamp >>", tm_stamp)
                    image_data = ech_tmstamp_obj.NextImg.imageData
                    # Decode the image here
                    img_np_arr = np.fromstring(image_data, np.uint8)
                    # print("img_np_arr :", img_np_arr)
                    decoded_img_arr = cv2.imdecode(img_np_arr, cv2.IMREAD_UNCHANGED)

                    # initial_bbox = [xmin, ymin, xmax, ymax]
                    # initial_bbox = [int(box[0]), int(box[1]), int(box[0]) + int(box[2]), int(box[1]) + int(box[3])]

                    for evry_attrib in ech_tmstamp_obj.CurrentAttr:
                        trackid = evry_attrib.trackID
                        tracker_name = evry_attrib.trackername
                        if tracker_name == 'Re3':
                            if evry_attrib.hasUserCorrected == 3 or evry_attrib.hasUserCorrected == 1:
                                manually_corrected = True
                            else:
                                manually_corrected = False
                            print("manually_corrected ::>", manually_corrected)
                            box_lst = []
                            for evry_ordinate_set in evry_attrib.ROI:
                                x_ordinate = evry_ordinate_set.X
                                box_lst.append(x_ordinate)
                                y_ordinate = evry_ordinate_set.Y
                                box_lst.append(y_ordinate)
                            # print("box_lst ::>", box_lst)
                            # name = 'pedastrian'
                            # imageRGB = img_np_arr

                            imageRGB_obj = Image.fromarray(np.uint8(decoded_img_arr)).convert('RGB')
                            imageRGB_arr = np.array(imageRGB_obj)
                            # imageRGB = imageRGB_arr
                            imageRGB = imageRGB_arr[:, :, ::-1]

                            # print("imageRGB>>", np.array(imageRGB))

                            # image_path = r'D:\Tanveer\code\github_sync\TrackID-generator\Adnet_tracker\0001.jpeg'
                            # image = cv2.imread(image_path)
                            # # Tracker expects RGB, but opencv loads BGR.
                            # imageRGB = image[:, :, ::-1]

                            # Uncomment the below line for testing
                            # bbox_coord_lst = box_lst
                            bbox_coord_lst = [box_lst[0], box_lst[1], box_lst[2], box_lst[5]]
                            print("non tracked box >>", bbox_coord_lst)
                            global PREV_IMAGE
                            if manually_corrected:
                                if PREV_IMAGE is not None:
                                    # if trackid not in self.trkid_lst:
                                    bbox, trackeddata = self.tracker.track(trackid, np.array(PREV_IMAGE), bbox_coord_lst)
                                    bbox, trackeddata = self.tracker.track(trackid, np.array(imageRGB))
                                    bbox = [int(evry_ordinate) for evry_ordinate in bbox]
                                else:
                                    bbox, trackeddata = self.tracker.track(trackid, np.array(imageRGB), bbox_coord_lst)
                                    bbox, trackeddata = self.tracker.track(trackid, np.array(imageRGB))
                                    bbox = [int(evry_ordinate) for evry_ordinate in bbox]
                            else:
                                bbox, trackeddata = self.tracker.track(trackid, np.array(imageRGB))
                                bbox = [int(evry_ordinate) for evry_ordinate in bbox]
                            print("tracked box >>", bbox)
                            print("\n")
                            PREV_IMAGE = imageRGB
                            multi_track_bbox_dict[trackid] = list(bbox)
                        self.publish_tracked_data(multi_track_bbox_dict)

    def inform_tracker_ready(self):
        # Inform model is loaded
        # time.sleep(2)

        lbl_response_obj = algointerface_pb2.AlgoState()
        lbl_response_obj.isReady = True
        self.lt5_algo_publr_obj.send(lbl_response_obj.SerializeToString())

    def abort_algo(self, topic_name, msg, time):

        global BEXIT
        BEXIT = False

    def define_subscr_callbacks(self):

        # For Image data
        self.lt5_img_subscr_obj.set_callback(self.predict_tracker_result)
        self.lt5_finl_subscr_obj.set_callback(self.abort_algo)
        while ecal.ok() and BEXIT:
            # print("#########################", BEXIT)
            time.sleep(0.1)
            if ALGO_READINESS:
                self.inform_tracker_ready()

if __name__ == '__main__':

    Re3TrackerLT5()
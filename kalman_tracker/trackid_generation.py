

"""
Run the detector. Publish images from LT5G tool.
Detector subscribes and generates BBoxes.
The ROIs are written into labeled output JSON file of the LT5G tool
The same JSON file serves as an input to the tracker.
The track IDs for each box is generated.
Rewrite the JSON file updating the track ID.
Compare the detector and tracker output.
"""

import json
import cv2
import conti_cv_py_detection.tracking.sort_tracker as st
import conti_cv_py_utilities.bounding_box as bb
import numpy as np
from PIL import Image
from PIL import ImageDraw
from os import listdir, getcwd
from os.path import isfile, join
import sys

CONFIG_JSON = "config.json"

class GenerateTrackIDsForDetector(object):

    def __init__(self):

        try:
            with open(LABELED_OUTPUT_JSON) as data_file:
                self.lbld_json_obj = json.load(data_file)
        except FileNotFoundError as e:
            print("Labeled output JSON file not found " + str(e))
            exit(0)
        with open(CONFIG_JSON) as conf_handle:
            config_json= json.load(conf_handle)
        self.retain_original_coordinates = True if config_json['tracker']["retain_original_coordinates"] == "True" else False
        parameters = []
        self.s_tracker = st.SortTracker(parameters, max_age=config_json['tracker']['max_age'],
                                        min_hits=config_json['tracker']['min_hits'])

        self.generate_BBox_obj()


    def generate_BBox_obj(self):

        # Make a copy of the labeled output data
        from copy import deepcopy
        self.lbld_json_op_copy = deepcopy(self.lbld_json_obj)

        img_flnames_lst = self.get_img_filenames(IMAGE_FOLDER_PATH)
        # Read each frame labeled data
        anno_elem_lst = self.lbld_json_op_copy['Sequence'][0]['DeviceData'][0]['ChannelData'][0]['AnnotatedElements']

        # print("anno_elem_lst :: ", anno_elem_lst)
        for idx, ech_frame in enumerate(anno_elem_lst):
            tm_stamp = ech_frame['TimeStamp']
            print("tm_stamp >> ", tm_stamp)
            frame_num = ech_frame['FrameNumber']
            obj_details_lst = ech_frame['FrameAnnoElements']
            # BBox list to be given to tracker
            detection_rois = []
            for ech_obj_det in obj_details_lst:
                trackid = ech_obj_det['Trackid']

                class_name = ech_obj_det['category']
                if ech_obj_det['shape']['x']:
                    # print("trackid >> ", trackid)
                    xmin = ech_obj_det['shape']['x'][0]
                    xmax = ech_obj_det['shape']['x'][1]
                    ymin = ech_obj_det['shape']['y'][0]
                    ymax = ech_obj_det['shape']['y'][1]

                    if xmin != -1:

                        detection_rois.append(bb.BoundingBox((xmin), (ymin),(xmax), (ymax),
                                               class_name=class_name))
                        # detection_rois.append(bb.BoundingBox((xmin), (ymin), (xmax), (ymax),
                        #                                      class_name=class_name, track_id=trackid))

            # Read the image to be fed to the tracker
            img_array = self.get_image_for_this_frame(img_names_lst=img_flnames_lst, timestamp=tm_stamp)
            tracks = self.s_tracker.improveBBoxes(img_array, detection_rois, frame_num)
            # self.display_tracked_objects(img_array, tracks)

            self.update_trackids_in_json(tm_stamp, tracks, ech_frame, detection_rois)

        # Write into JSON file
        with open(OUTPUT_JSON_TRACKED, 'w') as outfile:
            json.dump(self.lbld_json_op_copy, outfile)

    def display_tracked_objects(self, img_array, tracks):
        frame_trackID_det_lst = []
        for k, BBoxList in tracks.__dict__.items():
            # Track IDs are arranged in descending order. Reverse the order
            for ech_BBoxObj in BBoxList[::-1]:
                trckd_ordinate = ech_BBoxObj.json_content()
                # print("trckd_ordinate :> ", trckd_ordinate)
                frame_trackID_det_lst.append(trckd_ordinate)
        # print("frame_trackID_det_lst :: ", frame_trackID_det_lst)
        for ech_track_details in frame_trackID_det_lst:
            image_rgb = Image.fromarray(np.uint8(img_array)).convert('RGB')
            # image_rgb = img_array
            draw = ImageDraw.Draw(image_rgb)
            (left, right, top, bottom) = [ech_track_details['x1'],
            ech_track_details['x2'],
            ech_track_details['y1'],
            ech_track_details['y2']]
            draw.line([(left, top), (left, bottom), (right, bottom),
                       (right, top), (left, top)], width=2, fill='red')
            draw.text((int(left), int(top)), str(ech_track_details['track_id']), font=None)
            np.copyto(img_array, np.array(image_rgb))
            # ech_track_details['track_id']
        # Display the decoded image to verify if it has been decoded
        cv2.imwrite('color_img.jpg', img_array)
        cv2.imshow('Color image', img_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def update_trackids_in_json(self, tm_stamp, tracks, ech_frame_json_dict, original_rois_lst):

        # Validate the timeframe so that track ids are updated for correct timestamp
        if tm_stamp == ech_frame_json_dict['TimeStamp']:
            # print(">>=", ech_frame_json_dict)

            # Read the generated track IDs and the update the existing JSON/ Create a new JSON
            frame_trackID_det_lst = []
            for k, BBoxList in tracks.__dict__.items():
                # Track IDs are arranged in descending order. Reverse the order
                for ech_BBoxObj in BBoxList[::-1]:
                    trckd_ordinate = ech_BBoxObj.json_content()
                    # print("trckd_ordinate :> ", trckd_ordinate)
                    if not self.retain_original_coordinates:
                        frame_trackID_det_lst.append(trckd_ordinate)
                    else:
                        # Compute the iou for each tracked box with the original box supplied with the frame
                        # If the iou is gteater than 0.85 or 0.9 then assign the coordinates with the trackID
                        for evry_orig_BBox_obj in original_rois_lst:
                            orig_box_dict = evry_orig_BBox_obj.json_content()
                            # print("orig_box_dict>:", orig_box_dict)
                            iou_num = evry_orig_BBox_obj.iou(ech_BBoxObj)
                            # print("iou::", iou_num)
                            if iou_num > 0.85:
                                orig_box_dict['track_id'] = trckd_ordinate['track_id']
                                frame_trackID_det_lst.append(orig_box_dict)
            # original_rois_lst = [evry_orig_BBox_obj.json_content() for evry_orig_BBox_obj in original_rois_lst]
            # print("original_rois_lst::", original_rois_lst, len(original_rois_lst))

            # Now iterate through the frame data to only update the track IDs
            # Check if x co ordinate list is empty or not

            print("frame_trackID_det_lst >> ", frame_trackID_det_lst, len(frame_trackID_det_lst))
            print("\n")

            ech_frame_json_dict['FrameAnnoElements'] = []
            for ech_track_details in frame_trackID_det_lst:
                # break
                track_ordinate_details = {
                    "Hierarchy": 0,
                    "Trackid": ech_track_details['track_id'],
                    "angle": 0,
                    "attributes": {},
                    "baseimage": "",
                    "category": ech_track_details['class_name'],
                    "combinedimage": "",
                    "height": ech_track_details['y2'] - ech_track_details['y1'],
                    "imagedata": "",
                    "imageheight": 0,
                    "imagename": "",
                    "imagetype": "",
                    "imagewidth": 0,
                    "keypoints": {
                    },
                    "shape": {
                        "Algo Generated": "YES",
                        "Manually Corrected": "NO",
                        "thickness": 0,
                        "type": "Box",
                        "x": [
                            ech_track_details['x1'],
                            ech_track_details['x2']
                        ],
                        "y": [
                            ech_track_details['y1'],
                            ech_track_details['y2']
                        ]
                    },
                    "width": ech_track_details['x2'] - ech_track_details['x1']

                }
                ech_frame_json_dict['FrameAnnoElements'].append(track_ordinate_details)

            # print("ech_frame_json_dict :: ", ech_frame_json_dict)
            # exit(0)


    def get_image_for_this_frame(self, img_names_lst, timestamp):

        file_name = [evry_flname for evry_flname in img_names_lst if str(timestamp) in evry_flname][0]

        file_path = join(IMAGE_FOLDER_PATH, file_name)
        # print("file_name :: ", file_path)
        img_arr = cv2.imread(file_path)
        # print("img_arr :>> ", img_arr.shape)
        return img_arr

    def get_img_filenames(self, img_dir):

        try:
            img_flnames_lst = [f for f in listdir(IMAGE_FOLDER_PATH) if isfile(join(IMAGE_FOLDER_PATH, f))]
        except FileNotFoundError as e:
            print("Image folder path not found " + str(e))
            img_flnames_lst = []
            exit(0)
        # print("img_flnames_lst :: ", img_flnames_lst)
        return img_flnames_lst

if __name__ == '__main__':
    # Win 7
    LABELED_OUTPUT_JSON = r'D:\work\code\LT5G\Ticket_folder\SR_BatchTicket_Latest\LabeledData\SR_BatchTicket_Latest_LabelData_coco_no_track.json'
    # LABELED_OUTPUT_JSON = r'SR_BatchTicket_Latest_LabelDatat.json'
    IMAGE_FOLDER_PATH = r'D:\work\code\LT5G\Ticket_folder\SR_BatchTicket_Latest\Images'

    # Win 10
    LABELED_OUTPUT_JSON = r'D:\Work\2018\code\LT5G\ticket_folders\SR_BatchTicket_Latest\LabeledData\SR_BatchTicket_Latest_LabelData.json'
    IMAGE_FOLDER_PATH = r'D:\Work\2018\code\LT5G\ticket_folders\SR_BatchTicket_Latest\Images'

    LABELED_OUTPUT_JSON = getcwd() + r'\Ticket_folder\LabeledData\Ticket_folder_LabelData_d.json'
    IMAGE_FOLDER_PATH = getcwd() + r'\Ticket_folder\Images'

    OUTPUT_JSON_TRACKED = r'Ticket_folder_LabelData.json'

    print("Argument 1 : LABELED_OUTPUT_JSON file path ex: D:\SR_BatchTicket_Latest\LabeledData\SR_BatchTicket_Latest_LabelData.json")
    print("Argument 2 : IMAGE_FOLDER_PATH image folder ex: D:\SR_BatchTicket_Latest\Images")
    print("Argument 3 : OUTPUT_JSON_TRACKED file name ex: D:\SR_BatchTicket_Latest_LabelData.json")
    LABELED_OUTPUT_JSON = sys.argv[1]
    IMAGE_FOLDER_PATH = sys.argv[2]
    OUTPUT_JSON_TRACKED = sys.argv[3]

    GenerateTrackIDsForDetector()
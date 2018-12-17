

import ecal
import AlgoInterface_pb2
import sys
import json
import time
import numpy as np
import cv2
import os
from PIL import Image
from PIL import ImageDraw
from datetime import datetime
# Tracker libraries
import tensorflow as tf
import commons
from boundingbox import BoundingBox, Coordinate
from configs import ADNetConf
from networks import ADNetwork
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
TOPICS_JSON = 'topics.json'
CONFIG_YAML = 'conf/repo.yaml'
BEXIT = True
ALGO_READINESS = True

DISPLAY_IMAGE_WITH_GTBOX = False

if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(__file__)
print("application_path ::", application_path)
topics_json_path = os.path.join(application_path, TOPICS_JSON)
conf_yaml_path = os.path.join(application_path, CONFIG_YAML)
print("conf_yaml_path ::", conf_yaml_path)

class AdNetTracker(object):
    MAX_BATCHSIZE = 512
    RUN_COUNTER = 0
    def __init__(self, tracker_request, tracker_response):

        # Topic names
        self.tracker_request = tracker_request
        self.tracker_response = tracker_response
        # Tracker initializers  './conf/repo.yaml'
        ADNetConf.get(conf_yaml_path)
        self.tensor_input = tf.placeholder(tf.float32, shape=(None, 112, 112, 3), name='patch')
        self.tensor_action_history = tf.placeholder(tf.float32, shape=(None, 1, 1, 110), name='action_history')
        self.tensor_lb_action = tf.placeholder(tf.int32, shape=(None,), name='lb_action')
        self.tensor_lb_class = tf.placeholder(tf.int32, shape=(None,), name='lb_class')
        self.tensor_is_training = tf.placeholder(tf.bool, name='is_training')
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')
        self.persistent_sess = tf.Session(config=tf.ConfigProto(
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1
        ))

        self.adnet = ADNetwork(self.learning_rate_placeholder)
        self.adnet.create_network(self.tensor_input, self.tensor_lb_action, self.tensor_lb_class,
                                  self.tensor_action_history, self.tensor_is_training)
        if 'ADNET_MODEL_PATH' in os.environ.keys():
            self.adnet.read_original_weights(self.persistent_sess, os.environ['ADNET_MODEL_PATH'])
        else:
            self.adnet.read_original_weights(self.persistent_sess)

        # print("self.action_histories >>", ADNetConf.get())
        self.action_histories = np.array([0] * ADNetConf.get()['action_history'], dtype=np.int8)
        self.action_histories_old = np.array([0] * ADNetConf.get()['action_history'], dtype=np.int8)
        self.histories = []
        self.iteration = 0
        self.imgwh = None

        self.callback_redetection = self.redetection_by_sampling

        print("Tracker initialization Done!!")

        # Initialize eCAL
        ecal.initialize(sys.argv, "object tracking")
        # Read the JSON files
        with open(topics_json_path) as data_file:
            self.json_data = json.load(data_file)
        # Define the callbacks for publisher subscriber
        self.initialize_subscr_topics()
        self.initialize_publsr_topics()

        # The callbacks will redirect to the tracker function and publish predicted ROI
        self.define_subscr_callbacks()

    def __del__(self):
        self.persistent_sess.close()

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

    def display_untracked_objs_img(self, decoded_img_arr, BBox_assorted_dict):

        for ech_trk_id, ech_bbox_lst in BBox_assorted_dict.items():
            # print("ech_trk_id", ech_trk_id)
            image_rgb = Image.fromarray(np.uint8(decoded_img_arr)).convert('RGB')
            # print("image_rgb>>", image_rgb)
            draw = ImageDraw.Draw(image_rgb)
            # ech_bbox_lst arranged in the format xmin, ymin, xmax, ymax
            (left, right, top, bottom) = [ech_bbox_lst[0], ech_bbox_lst[2], ech_bbox_lst[1], ech_bbox_lst[3]]
            print("??", (left, right, top, bottom))
            draw.line([(left, top), (left, bottom), (right, bottom),
                       (right, top), (left, top)], width=2, fill='red')
            draw.text((int(left), int(top)), str(ech_trk_id), font=None)
            np.copyto(decoded_img_arr, np.array(image_rgb))
        # Display the decoded image to verify if it has been decoded
        cv2.imwrite('color_img.jpg', decoded_img_arr)
        cv2.imshow('Color image', decoded_img_arr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def accumulate_ground_truth_boxes(self, trackid_bbox_dict):
        """

        :param trackid_bbox_dict: Dictionary containing key as track ID, value is a list having bounding box coordinates
        in the format (xmin, ymin, xmax, ymax)
        :return:
        """
        # boxes = []
        boxes = {}
        for ech_trkid, bbox_lst in trackid_bbox_dict.items():
            x = bbox_lst[0]
            y = bbox_lst[1]
            w = bbox_lst[2] - bbox_lst[0]
            h = bbox_lst[3] - bbox_lst[1]
            box = BoundingBox(x, y, w, h)
            # boxes.append(box)
            boxes[ech_trkid]=box
        return boxes

    def _get_features(self, samples):
        feats = []
        for batch in commons.chunker(samples, AdNetTracker.MAX_BATCHSIZE):
            feats_batch = self.persistent_sess.run(self.adnet.layer_feat, feed_dict={
                self.adnet.input_tensor: batch
            })
            feats.extend(feats_batch)
        return feats

    def _finetune_fc(self, img, pos_boxes, neg_boxes, pos_lb_action, learning_rate, iter, iter_score=1):
        BATCHSIZE = ADNetConf.g()['minibatch_size']

        def get_img(idx, posneg):
            if isinstance(img, tuple):
                return img[posneg][idx]
            return img

        pos_samples = [commons.extract_region(get_img(i, 0), box) for i, box in enumerate(pos_boxes)]
        neg_samples = [commons.extract_region(get_img(i, 1), box) for i, box in enumerate(neg_boxes)]
        # pos_feats, neg_feats = self._get_features(pos_samples), self._get_features(neg_samples)

        # commons.imshow_grid('pos', pos_samples[-50:], 10, 5)
        # commons.imshow_grid('neg', neg_samples[-50:], 10, 5)
        cv2.waitKey(1)

        for i in range(iter):
            batch_idxs = commons.random_idxs(len(pos_boxes), BATCHSIZE)
            batch_feats = [x.feat for x in commons.choices_by_idx(pos_boxes, batch_idxs)]
            batch_lb_action = commons.choices_by_idx(pos_lb_action, batch_idxs)
            self.persistent_sess.run(
                self.adnet.weighted_grads_op1,
                feed_dict={
                    self.adnet.layer_feat: batch_feats,
                    self.adnet.label_tensor: batch_lb_action,
                    self.adnet.action_history_tensor: np.zeros(shape=(BATCHSIZE, 1, 1, 110)),
                    self.learning_rate_placeholder: learning_rate,
                    self.tensor_is_training: True
                }
            )

            if i % iter_score == 0:
                # training score auxiliary(fc2)
                # -- hard score example mining
                scores = []
                for batch_neg in commons.chunker([x.feat for x in neg_boxes], AdNetTracker.MAX_BATCHSIZE):
                    scores_batch = self.persistent_sess.run(
                        self.adnet.layer_scores,
                        feed_dict={
                            self.adnet.layer_feat: batch_neg,
                            self.adnet.action_history_tensor: np.zeros(shape=(len(batch_neg), 1, 1, 110)),
                            self.learning_rate_placeholder: learning_rate,
                            self.tensor_is_training: False
                        }
                    )
                    scores.extend(scores_batch)
                desc_order_idx = [i[0] for i in sorted(enumerate(scores), reverse=True, key=lambda x:x[1][1])]

                # -- train
                batch_feats_neg = [x.feat for x in commons.choices_by_idx(neg_boxes, desc_order_idx[:BATCHSIZE])]
                self.persistent_sess.run(
                    self.adnet.weighted_grads_op2,
                    feed_dict={
                        self.adnet.layer_feat: batch_feats + batch_feats_neg,
                        self.adnet.class_tensor: [1]*len(batch_feats) + [0]*len(batch_feats_neg),
                        self.adnet.action_history_tensor: np.zeros(shape=(len(batch_feats)+len(batch_feats_neg), 1, 1, 110)),
                        self.learning_rate_placeholder: learning_rate,
                        self.tensor_is_training: True
                    }
                )

    def initial_finetune(self, img, detection_box):

        # print("Start initial_finetune1")
        # generate samples
        pos_num, neg_num = ADNetConf.g()['initial_finetune']['pos_num'], ADNetConf.g()['initial_finetune']['neg_num']
        # print("Ending initial_finetune1")
        pos_boxes, neg_boxes = detection_box.get_posneg_samples(self.imgwh, pos_num, neg_num, use_whole=True)
        # print("Ending initial_finetune133")
        pos_lb_action = BoundingBox.get_action_labels(pos_boxes, detection_box)
        # print("Ending initial_finetune44")


        feats = self._get_features([commons.extract_region(img, box) for i, box in enumerate(pos_boxes)])
        for box, feat in zip(pos_boxes, feats):
            box.feat = feat
        feats = self._get_features([commons.extract_region(img, box) for i, box in enumerate(neg_boxes)])
        for box, feat in zip(neg_boxes, feats):
            box.feat = feat

        # print("Ending initial_finetune2")
        # train_fc_finetune_hem
        self._finetune_fc(
            img, pos_boxes, neg_boxes, pos_lb_action,
            ADNetConf.get()['initial_finetune']['learning_rate'],
            ADNetConf.get()['initial_finetune']['iter']
        )

        self.histories.append((pos_boxes, neg_boxes, pos_lb_action, np.copy(img), self.iteration))
        # print("Ending initial_finetune")

    def tracking(self, img, curr_bbox, trackid):
        # print("---------------------tracking")
        self.iteration += 1
        is_tracked = True
        boxes = []
        self.latest_score = -1
        # self.stopwatch.start('tracking.do_action')
        for track_i in range(ADNetConf.get()['predict']['num_action']):
            patch = commons.extract_region(img, curr_bbox)

            # forward with image & action history
            actions, classes = self.persistent_sess.run(
                [self.adnet.layer_actions, self.adnet.layer_scores],
                feed_dict={
                    self.adnet.input_tensor: [patch],
                    self.adnet.action_history_tensor: [commons.onehot_flatten(self.action_histories)],
                    self.tensor_is_training: False
                }
            )

            latest_score = classes[0][1]
            if latest_score < ADNetConf.g()['predict']['thresh_fail']:
                is_tracked = False
                self.action_histories_old = np.copy(self.action_histories)
                self.action_histories = np.insert(self.action_histories, 0, 12)[:-1]
                break
            else:
                self.failed_cnt = 0
            self.latest_score = latest_score

            # move box
            action_idx = np.argmax(actions[0])
            self.action_histories = np.insert(self.action_histories, 0, action_idx)[:-1]
            prev_bbox = curr_bbox
            curr_bbox = curr_bbox.do_action(self.imgwh, action_idx)
            if action_idx != ADNetwork.ACTION_IDX_STOP:
                if prev_bbox == curr_bbox:
                    print('action idx', action_idx)
                    print(prev_bbox)
                    print(curr_bbox)
                    raise Exception('box not moved.')

            # oscillation check
            if action_idx != ADNetwork.ACTION_IDX_STOP and curr_bbox in boxes:
                action_idx = ADNetwork.ACTION_IDX_STOP

            if action_idx == ADNetwork.ACTION_IDX_STOP:
                break

            boxes.append(curr_bbox)
        #self.stopwatch.stop('tracking.do_action')
        # Do not read it from ground truth, read from eCAL
        # gt_boxes = BoundingBox.read_vid_gt('./data/BlurCar2/',idx)
        gt_boxes = self.accumulate_ground_truth_boxes(self.grnd_truth_dict)
        #self.stopwatch.start('total')
        #_logger.info('---- start dataset l=%d' % (len(gt_boxes)))

        # redetection when tracking failed
        new_score = 0.0
        # print("is_tracked>>", is_tracked)
        if not is_tracked:
            self.failed_cnt += 1
            print(self.failed_cnt)
            # run redetection callback function
            # Commenting the below line so that the current box assigned is the ground truth box of the previous frame
            # curr_bbox=gt_boxes[idx]
            new_box, new_score = self.callback_redetection(curr_bbox, img)
            if new_box is not None:
                curr_bbox = new_box
                patch = commons.extract_region(img, curr_bbox)
            # _logger.debug('redetection success=%s' % (str(new_box is not None)))

        # save samples
        if is_tracked or new_score > ADNetConf.g()['predict']['thresh_success']:
            # self.stopwatch.start('tracking.save_samples.roi')
            imgwh = Coordinate.get_imgwh(img)
            pos_num, neg_num = ADNetConf.g()['finetune']['pos_num'], ADNetConf.g()['finetune']['neg_num']
            pos_boxes, neg_boxes = curr_bbox.get_posneg_samples(
                imgwh, pos_num, neg_num, use_whole=False,
                pos_thresh=ADNetConf.g()['finetune']['pos_thresh'],
                neg_thresh=ADNetConf.g()['finetune']['neg_thresh'],
                uniform_translation_f=2,
                uniform_scale_f=5
            )
            # self.stopwatch.stop('tracking.save_samples.roi')
            # self.stopwatch.start('tracking.save_samples.feat')
            feats = self._get_features([commons.extract_region(img, box) for i, box in enumerate(pos_boxes)])
            for box, feat in zip(pos_boxes, feats):
                box.feat = feat
            feats = self._get_features([commons.extract_region(img, box) for i, box in enumerate(neg_boxes)])
            for box, feat in zip(neg_boxes, feats):
                box.feat = feat
            pos_lb_action = BoundingBox.get_action_labels(pos_boxes, curr_bbox)
            self.histories.append((pos_boxes, neg_boxes, pos_lb_action, np.copy(img), self.iteration))

            # clear old ones
            self.histories = self.histories[-ADNetConf.g()['finetune']['long_term']:]
        # online finetune
        if self.iteration % ADNetConf.g()['finetune']['interval'] == 0 or is_tracked is False:
            img_pos, img_neg = [], []
            pos_boxes, neg_boxes, pos_lb_action = [], [], []
            pos_term = 'long_term' if is_tracked else 'short_term'
            for i in range(ADNetConf.g()['finetune'][pos_term]):
                if i >= len(self.histories):
                    break
                pos_boxes.extend(self.histories[-(i+1)][0])
                pos_lb_action.extend(self.histories[-(i+1)][2])
                img_pos.extend([self.histories[-(i+1)][3]]*len(self.histories[-(i+1)][0]))
            for i in range(ADNetConf.g()['finetune']['short_term']):
                if i >= len(self.histories):
                    break
                neg_boxes.extend(self.histories[-(i+1)][1])
                img_neg.extend([self.histories[-(i+1)][3]]*len(self.histories[-(i+1)][1]))
            self._finetune_fc(
                (img_pos, img_neg), pos_boxes, neg_boxes, pos_lb_action,
                ADNetConf.get()['finetune']['learning_rate'],
                ADNetConf.get()['finetune']['iter']
            )

        # cv2.imshow('patch', patch)
        # return curr_bbox
        return {trackid: curr_bbox}

    def redetection_by_sampling(self, prev_box, img):
        """
        default redetection method
        """
        imgwh = Coordinate.get_imgwh(img)
        translation_f = min(1.5, 0.6 * 1.15**self.failed_cnt)
        candidates = prev_box.gen_noise_samples(imgwh, 'gaussian', ADNetConf.g()['redetection']['samples'],
                                                gaussian_translation_f=translation_f)

        scores = []
        for c_batch in commons.chunker(candidates, AdNetTracker.MAX_BATCHSIZE):
            samples = [commons.extract_region(img, box) for box in c_batch]
            classes = self.persistent_sess.run(
                self.adnet.layer_scores,
                feed_dict={
                    self.adnet.input_tensor: samples,
                    self.adnet.action_history_tensor: [commons.onehot_flatten(self.action_histories_old)]*len(c_batch),
                    self.tensor_is_training: False
                }
            )
            scores.extend([x[1] for x in classes])
        top5_idx = [i[0] for i in sorted(enumerate(scores), reverse=True, key=lambda x: x[1])][:5]
        mean_score = sum([scores[x] for x in top5_idx]) / 5.0
        if mean_score >= self.latest_score:
            mean_box = candidates[0]
            for i in range(1, 5):
                mean_box += candidates[i]
            return mean_box / 5.0, mean_score
        return None, 0.0

    def run_tracker(self, decoded_img_arr, BBox_assorted_dict, manually_corrected):

        # Convert into Bounding box object
        # Bounding box object will have xmin, ymin, width and height
        # Derive width and height from the 4-coordinates
        # gt_boxes must have list of Bunding box objects
        gt_boxes_dict = self.accumulate_ground_truth_boxes(BBox_assorted_dict)
        # initialization : initial fine-tuning
        # Multiple trackid and BBox association
        multi_track_bbox = {}
        # AdNetTracker.RUN_COUNTER = 0
        for trackid, gt_box in gt_boxes_dict.items():
            st_time = datetime.now()
            print("trackid>>", trackid)
            curr_bbox = gt_box
            self.imgwh = Coordinate.get_imgwh(decoded_img_arr)
            print("self.imgwh >>", self.imgwh)
            if AdNetTracker.RUN_COUNTER == 0 or manually_corrected:
                self.initial_finetune(decoded_img_arr, gt_box)
            AdNetTracker.RUN_COUNTER += 1

            # predicted_box = self.tracking(decoded_img_arr, curr_bbox, trackid)
            # print("predicted_box>>", predicted_box, predicted_box.xy.x, predicted_box.xy.y)
            print("previous box::", curr_bbox)
            # predicted_box_dict contains key as track ID and value is BBox object
            predicted_box_dict = self.tracking(decoded_img_arr, curr_bbox, trackid)
            print("predicted_box_dict::", predicted_box_dict)
            ed_time = datetime.now()
            duration = ed_time - st_time
            print("tracking done in...", duration)
            multi_track_bbox.update(predicted_box_dict)
        # print("multi_track_bbox>>", multi_track_bbox)
        return multi_track_bbox

    def publish_tracked_data(self, multi_track_bbox_dict):
        if multi_track_bbox_dict:
            lbl_response_obj = AlgoInterface_pb2.LabelResponse()
            for ech_trk_id, bbox_obj in multi_track_bbox_dict.items():
                # print("nn>", ech_trk_id, bbox_obj)
                # print(dir(bbox_obj), dir(bbox_obj.wh))
                # print("pred boxes:", bbox_obj.wh.x, bbox_obj.wh.y)
                attrib_typ_obj = lbl_response_obj.NextAttr.add()
                attrib_typ_obj.trackID = ech_trk_id
                attrib_typ_obj.hasUserCorrected = 0
                attrib_typ_obj.trackername = self.json_data["tracker"]
                print("publish tracker name ::", attrib_typ_obj.trackername)
                # bbox_obj in format x,y,w,h
                xmin = bbox_obj.xy.x
                ymin = bbox_obj.xy.y
                xmax = bbox_obj.xy.x + bbox_obj.wh.x
                ymax = bbox_obj.xy.y + bbox_obj.wh.y
                print("coordinates::", xmin, ymin, xmax, ymax)

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
        lbl_request_obj = AlgoInterface_pb2.LabelRequest()
        if msg is not None:
            lbl_request_obj.ParseFromString(msg)
            image_data = lbl_request_obj.NextImg.imageData
            # Get the list of BBox within a list
            BBox_assorted_dict = {}
            for evry_attrib in lbl_request_obj.CurrentAttr:
                trackid = evry_attrib.trackID
                print("\ntrackid ::", trackid)
                if evry_attrib.trackername == self.json_data["tracker"]:
                    # If the variable is zero then it was not manually corrected, if more than zero then it was corrected
                    if evry_attrib.hasUserCorrected > 0:
                        manually_corrected = True
                    else:
                        manually_corrected = False
                    print("user corrected::", manually_corrected)
                    # BBox in list of format [xmin, ymin, xmax, ymax]
                    box_lst = []
                    for evry_ordinate_set in evry_attrib.ROI:
                        x_ordinate = evry_ordinate_set.X
                        box_lst.append(x_ordinate)
                        y_ordinate = evry_ordinate_set.Y
                        box_lst.append(y_ordinate)
                        # print("ordinate>>", x_ordinate, y_ordinate)
                    # print("box_lst ::", box_lst)
                    # BBox_assorted_dict[trackid] = box_lst

                    box_ordinate_lst = [box_lst[0], box_lst[1], box_lst[2], box_lst[5]]
                    BBox_assorted_dict[trackid] = box_ordinate_lst
                else:
                    print("Tracker name is " + self.json_data["tracker"])
                    print("But sending in " + evry_attrib.trackername)
            print("BBox_assorted_dict ::", BBox_assorted_dict)
            self.grnd_truth_dict = BBox_assorted_dict
            # Display the recieved image with bounding boxes
            # Decode the image here
            img_np_arr = np.fromstring(image_data, np.uint8)
            print("img_np_arr :", img_np_arr)
            decoded_img_arr = cv2.imdecode(img_np_arr, cv2.IMREAD_UNCHANGED)
            # cv2.imwrite('color_img.jpg', decoded_img_arr)
            # print("display_untracked_objs_img>>", decoded_img_arr)
            if DISPLAY_IMAGE_WITH_GTBOX:
                self.display_untracked_objs_img(decoded_img_arr, BBox_assorted_dict)
            multi_track_bbox_dict = self.run_tracker(decoded_img_arr, BBox_assorted_dict, manually_corrected)
            self.publish_tracked_data(multi_track_bbox_dict)

    def abort_algo(self, topic_name, msg, time):

        if topic_name == self.json_data['algo_end_response']:
            global BEXIT
            BEXIT = False

    def inform_tracker_ready(self):
        # Inform model is loaded
        # time.sleep(2)

        lbl_response_obj = AlgoInterface_pb2.AlgoState()
        lbl_response_obj.isReady = True
        self.lt5_algo_publr_obj.send(lbl_response_obj.SerializeToString())

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

    tracker_request = "SR_Request"
    tracker_response = "SR_Response"

    # tracker_request = sys.argv[1]
    # tracker_response = sys.argv[2]

    AdNetTracker(tracker_request, tracker_response)

# C:\Users\uidr8549\Envs\tracker-tf-cpu-py35\Scripts\pyinstaller.exe lt5_dl_tracker.py
# C:\Users\uidr8549\AppData\Local\Continuum\Anaconda3\Scripts\pyinstaller.exe lt5_dl_tracker.py
# C:\Users\uidr8549\AppData\Local\Continuum\Anaconda3\Scripts\pyinstaller.exe --onefile lt5_dl_tracker.py --add-data _ecal_py_3_5_x64.pyd;.
# _ecal_py_3_5_x64.pyd conf models topics.json - copy into the dist folder

# C:\Users\uidr8549\AppData\Local\Continuum\Anaconda3\python.exe lt5_dl_tracker.py SR_Request SR_Response



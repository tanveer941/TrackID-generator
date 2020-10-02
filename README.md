# TrackID-generator
Feed a set of images and their respective annotations for the objects to generate track IDs for the objects already detected
# Kalman tracker
This is a sort tracker based on Kalman filter. The tracker produces active and inactive tracks(co-ordinates) for every frame
comparing the co-ordinates from previous and current frame.

1. The pre-requisite being co-ordinates are generated for the objects for every frame by running a detector on it.
2. The sort tracker will assign a track ID for every object in the frame. In the subsequent frame it will confirm if it is the 
   same object as the previous one and assign the same track ID.
3. If an occlusion occurs on the object and it reappears, a new track ID is assigned to it.
4. Install the dependencies:	
	conti_cv_py_detection	
	conti_cv_py_utilities		 

## How to use? 
- Run the LabelTool5G in batch mode. The labeled data file is generated.
- LabeledData file contains co-ordinates for objects detected for every frame.
- Run the trackid_generation.py file. The input to the file is the LabeledData JSON and the folder containing its corresponding images.
- Specify the output path for the JSON having the updated track IDs. 
- Sample labeled data JSON and the folder containing images are in the same repo.
- Execute **trackid_generation.py \Ticket_folder\LabeledData\Ticket_folder_LabelData.json Ticket_folder LabeledData_output_tracked.json**

# Adnet(Action-Decision Network) Tracker
It makes use of Deep reinforcement learning. It has a pre-trained model with the saved weights for feature extraction.

1. The input to the tracker is current image, previous image co-ordinates and the extracted features of the image enclosed within the bounding box.
2. The Label tool publishes the image and the coordinates to the tracker along with an indicator to show if it has been user corrected.
3. The label for the first frame by default will be fine tuned and also for the ones which are manually corrected.
4. Fine tuning also occurs for every 30th frame and also when also when the confidence level for the features extracted is less than the threshold 
	as defined in the configuration file.
5. Fine tuning takes a considerable amount of time(~20-30seconds).
6. The intermediate frames takes approx. a second to produce the bounding box.
7. Link to the original github repository is https://github.com/ildoonet/tf-adnet-tracking 

## How to use?
- Run the file **lt5_dl_tracker.py**
- The application is now ready to subscribe for the messages(images and coordinates)
- The result is a publisher producing predicted bounding box

## Creating Executable 
- \Local\Continuum\Anaconda3\Scripts\pyinstaller.exe --onefile lt5_dl_tracker.py --add-data       _ecal_py_3_5_x64.pyd;.
- Copy the folders 'conf', 'models' and the file 'topics.json' into the place of the executable

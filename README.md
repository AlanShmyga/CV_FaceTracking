# CV_FaceTracking
Project for Facial Recognition and Tracking. Diploma project for "Robot Dreams" school Computer Vision course (early 2023).

### Project Description
Create a script that combines face detection and face recognition DNN models.
The models should detect any face in real time from the default system camera 
but recognize and track only the face that is set as target.

### Instruction
* The target face image path can be passed to the script as an optional "--image1" (or "-i1") run argument.
* The face detection model and face recognition models needs to be downloaded before the script run.
Paths to face detection and face recognition models can be passed as optional arguments
"--face_detection_model" (or "-fd") and
"--face_recognition_model" (or "-fr") accordingly

* To run the script int your terminal go to the script containing folder on your file system
and run "python3 DetectAndTrackFace.py" command.

* To stop script execution press "q" on keyboard.
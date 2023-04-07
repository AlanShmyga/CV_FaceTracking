#!/usr/bin/env python3
"""

This script runs OpenCV YuNet face detection model and SFace face recognition model to recognize the target face.
The script runs on live video capture from computer camera.

The target face image path can be passed to the script as an optional "--image1" (or "-i1") run argument.

The face detection model and face recognition models needs to be downloaded before the script run.
Paths to face detection and face recognition models can be passed as optional arguments
"--face_detection_model" (or "-fd") and
"--face_recognition_model" (or "-fr") accordingly

To run the script int your terminal go to the script containing folder on your file system
and run "python3 DetectAndTrackFace.py" command.

To stop script execution press "q" on keyboard.

@author: alan shmyha
"""

import argparse
import sys
import cv2
import numpy as np

DETECTED_IMG_LOG_MSG = "Face {}, top-left coordinates: " \
    "({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}"

parser = argparse.ArgumentParser()
parser.add_argument("--image1", "-i1", type=str, default="../data/IMG_4387.jpeg",
                    help="Path to the image containing the seeking face.")

parser.add_argument("--target_face_idx", "-tfi", type=int, default=1,
                    help="Index of the face on the source image.")

parser.add_argument("--face_detection_model", "-fd", type=str,
                    default="../opencv_zoo/models/face_detection_yunet/face_detection_yunet_2022mar.onnx",
                    help="Path to the face detection model. "
                         "Download the model at "
                         "https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet")

parser.add_argument("--face_recognition_model", "-fr", type=str,
                    default="../opencv_zoo/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
                    help="Path to the face recognition model. "
                         "Download the model at "
                         "https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface")
args = parser.parse_args()


def detect_and_recognize(img, src_feat):
    """Performs face detection and recognition on the target image.
    """
    # Detect faces on the current frame if any are present
    face_candidates = detector.detect(img)
    if face_candidates[1] is None:
        return

    # Align faces on the frame
    aligned_faces = [recognizer.alignCrop(img, face) for face in face_candidates[1]]

    # Extract features
    dst_feats = [recognizer.feature(face) for face in aligned_faces]

    # Define scores array to collect all the recognition scores
    scores = []
    # Go through all the faces and find the one we're seeking for
    for dst_feat in dst_feats:
        # Get the face recognition score and save it to the scores array under the corresponding index
        recognition_score = recognizer.match(src_feat, dst_feat, cv2.FaceRecognizerSF_FR_COSINE)
        scores.append(recognition_score)

    # Pick top scored recognized faces and mark them by the red rectangle in the frame
    match = face_candidates[1][np.argmax(scores)]
    x, y, w, h = [int(val) for val in match][:4]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=5)
    return x, y, w, h


if __name__ == "__main__":
    # Define source image for face recognition
    source_image = cv2.imread(args.image1)
    source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)

    # Create detector
    detector = cv2.FaceDetectorYN.create(args.face_detection_model, "", (320, 320))

    # Create recognizer
    recognizer = cv2.FaceRecognizerSF.create(args.face_recognition_model, "")

    # Create tracker
    tracker = cv2.TrackerMIL_create()

    # Define a video capture object
    cap = cv2.VideoCapture(0)

    # Set the detector size to the source image size
    width, height, _ = source_image.shape
    detector.setInputSize((height, width))

    # Detect the target face on the source image
    face_src = detector.detect(source_image)

    # We should know in advance which face on the source picture are we looking for
    target_face = face_src[1][args.target_face_idx]

    # Align
    face_src_aligned = recognizer.alignCrop(source_image, target_face)

    # Extract features
    feat_src = recognizer.feature(face_src_aligned)

    # Set desired frame size and set it to detector
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])

    # Define frame counter and the threshold for the recognizer application
    # Generally We'd like to apply the recognizer not on every frame to save the resources
    frame_count = 0
    frame_threshold = 5

    # Get through first frames to try to find the face (it might not be present on the fist image)
    for _ in range(5):
        ret, frame = cap.read()
        if not ret:
            sys.exit("Cannot read first frames")

        bbox = detect_and_recognize(frame, feat_src)
        if bbox:
            break

    if not bbox:
        sys.exit("No faces found on the initial frames")

    frame = cv2.resize(frame, (frameWidth, frameHeight))

    tracker.init(frame, bbox)

    while cap.isOpened():

        # Capture the video frame by frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, (frameWidth, frameHeight))

        # Wait for Face to get back if is temporarily out of the frame
        while cap.isOpened() and not bbox:
            ret, frame = cap.read()

            # Get the target Face coordinates if exists
            bbox = detect_and_recognize(frame, feat_src)

        ok, bbox = tracker.update(frame)
        x, y, w, h = bbox

        # Visualize Tracker
        center = (int((x + w / 2)), int((y + h / 2)))
        cv2.circle(frame, center, bbox[3] // 2, (0, 255, 255), thickness=2)

        # Visualize results
        cv2.imshow("Show Time", frame)

        # The 'q' button is set as the quitting button
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # After the loop release the cap object and destroy all the windows
    cap.release()
    cv2.destroyAllWindows()

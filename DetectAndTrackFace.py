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


def visualize(target_img, detected_faces, thickness=2):
    """
    Visualization function.

    Draws surrounding boxes and facial landmarks.
    """
    if detected_faces[1] is None:
        return
    for idx, face in enumerate(detected_faces[1]):
        print(DETECTED_IMG_LOG_MSG.format(idx, face[0], face[1], face[2], face[3], face[-1]))
        coords = face[:-1].astype(np.int32)
        cv2.rectangle(target_img, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0),
                      thickness)
        cv2.circle(target_img, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
        cv2.circle(target_img, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
        cv2.circle(target_img, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
        cv2.circle(target_img, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
        cv2.circle(target_img, (coords[12], coords[13]), 2, (0, 255, 255), thickness)


if __name__ == "__main__":
    # Define source image for face recognition
    source_image = cv2.imread(args.image1)
    source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)

    # Create the detector
    detector = cv2.FaceDetectorYN.create(args.face_detection_model, "", (320, 320))

    # Create the recognizer
    recognizer = cv2.FaceRecognizerSF.create(args.face_recognition_model, "")

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

    while cap.isOpened():

        # Capture the video frame by frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, (frameWidth, frameHeight))

        # Detect faces on the current frame if any are present
        faces = detector.detect(frame)

        # If this frame is the one where we should run the recognizer then let's do it
        if frame_count == frame_threshold:
            # Align faces on the frame
            aligned_faces = [recognizer.alignCrop(frame, face) for face in faces[1]]

            # Extract features
            feats_dst = [recognizer.feature(face) for face in aligned_faces]

            # Define scores array to collect all the recognition scores
            scores = []
            # Go through all the faces and find the one we're seeking for
            for feat_dst in feats_dst:
                # Get the face recognition score and save it to the scores array under the corresponding index
                recognition_score = recognizer.match(feat_src, feat_dst, cv2.FaceRecognizerSF_FR_COSINE)
                scores.append(recognition_score)

            # Pick top scored recognized faces and mark them by the red rectangle in the frame
            match = faces[1][np.argmax(scores)]
            x1, y1, w, h = match[0:4]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=5)
            # Do not forget to renew the frame counter
            frame_count = 0
        else:
            frame_count += 1

        # Draw rectangle around each face on the input image
        visualize(frame, faces)

        # Visualize results
        cv2.imshow("Live", frame)

        # The 'q' button is set as the quitting button
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # After the loop release the cap object and destroy all the windows
    cap.release()
    cv2.destroyAllWindows()

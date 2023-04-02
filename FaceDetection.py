#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script runs OpenCV YuNet face detection model on live video capture from computer camera.

@author: alan shmyha
"""

# import the opencv library
import cv2
import numpy as np


def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            print(
                'Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(
                    idx, face[0], face[1], face[2], face[3], face[-1]))
            coords = face[:-1].astype(np.int32)
            cv2.rectangle(input, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0),
                         thickness)
            cv2.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv2.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv2.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv2.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv2.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv2.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


if __name__ == '__main__':

    # Define source image for face recognition
    source_image = cv2.imread("../data/IMG_4387.jpeg")
    source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)

    # Create the detector
    detector = cv2.FaceDetectorYN.create(
        "../opencv_zoo/models/face_detection_yunet/face_detection_yunet_2022mar.onnx",
        "",
        (320, 320)
    )

    # Create the recognizer
    recognizer = cv2.FaceRecognizerSF.create(
        "../opencv_zoo/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        ""
    )

    tm = cv2.TickMeter()

    # Define a video capture object
    cap = cv2.VideoCapture(0)

    # Set the detector size to the source image size
    width, height, _ = source_image.shape
    detector.setInputSize((height, width))

    # Detect the target face on the source image
    face_src = detector.detect(source_image)

    # We know in advance that our target face is the second face on the picture so set it as desired
    target_face = face_src[1][1]

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

    while(cap.isOpened()):

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
        visualize(frame, faces, tm.getFPS())
        # Visualize results
        cv2.imshow('Live', frame)

        # The 'q' button is set as the quitting button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object and destroy all the windows
    cap.release()
    cv2.destroyAllWindows()

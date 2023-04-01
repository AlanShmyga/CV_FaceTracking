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
    scale = 1.0
    detector = cv2.FaceDetectorYN.create(
        "../opencv_zoo/models/face_detection_yunet/face_detection_yunet_2022mar.onnx",
        "",
        (320, 320),
        0.9,
        0.3,
        5000
    )
    # Define a video capture object
    tm = cv2.TickMeter()
    cap = cv2.VideoCapture(0)
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    detector.setInputSize([frameWidth, frameHeight])

    while(cap.isOpened()):
        # Capture the video frame by frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, (frameWidth, frameHeight))
        faces = detector.detect(frame)

        # Draw rectangle around each face
        # Draw results on the input image
        visualize(frame, faces, tm.getFPS())
        # Visualize results
        cv2.imshow('Live', frame)

        # The 'q' button is set as the quitting button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object and destroy all the windows
    cap.release()
    cv2.destroyAllWindows()

import cv2 as cv
import csv
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera.camera import get_frame, release
from gestures.gesture_model import get_landmarks
from render.renderer import draw_landmarks

# Create new csv file to write data to
csv_file = open('gesture_data.csv', 'a', newline='')
writer = csv.writer(csv_file)


while True:
    frame = get_frame()
    if frame is None:
        continue


    frame = cv.flip(frame, 1)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = get_landmarks(frame_rgb)
    draw_landmarks(frame, results.hand_landmarks)

    if results.hand_landmarks:
        hand = results.hand_landmarks[0]
        landmarks_flat = []
        for lm in hand:
            landmarks_flat.extend([lm.x, lm.y])

        key = cv.waitKey(1)
        if key == ord('f'):
            writer.writerow(landmarks_flat + ['fist'])
        elif key == ord('o'):
            writer.writerow(landmarks_flat + ['open'])
        elif key == ord('p'):
            writer.writerow(landmarks_flat + ['point'])
        elif key == ord('v'):
            writer.writerow(landmarks_flat + ['peace'])
        elif key == ord('q'):
            break
    else:
        key = cv.waitKey(1)
        if key == ord('q'):
            break

    cv.imshow('frame', frame)

release()
csv_file.close()

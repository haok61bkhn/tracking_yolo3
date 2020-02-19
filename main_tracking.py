import cv2
from PIL import Image
import copy
from tracker import Tracker
import numpy as np
# from keras import backend as K
from timeit import default_timer as timer
from detect_motor import *

ix, iy, ex, ey = -1, -1, -1, -1
cap_from_stream = False
path = 'street_tdh.mp4'
#path = 'rtsp://192.168.10.16:554'


def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))


def draw_rec(event, x, y, flags, param):
    global ix, iy, ex, ey, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        ex, ey = x, y
        cv2.rectangle(param, (ix, iy), (x, y), (0, 255, 0), 0)


def get_crop_size(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if cap_from_stream:
            frame = cv2.resize(frame, (1280, 720))
        cv2.namedWindow('draw_rectangle')
        cv2.setMouseCallback('draw_rectangle', draw_rec, frame)
        print("Choose your area of interest!")
        while 1:
            cv2.imshow('draw_rectangle', frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('a'):
                cv2.destroyAllWindows()
                break
        break


def main():
    # limit_mem()
    # Choose area of interest
    get_crop_size(path)
    print('Your area of interest: ', ix, ' ', iy, ' ', ex, ' ', ey)
    # area = (ix, iy, ex, ey)

    # Create opencv video capture object
    cap = cv2.VideoCapture(path)
    w = int(cap.get(3))
    h = int(cap.get(4))
    if cap_from_stream:
        w = 1280
        h = 720
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('res.avi', fourcc, 15, (w, h))

    # Create Object Detector
    detector = detector1()

    # Create Object Tracker
    tracker = Tracker(iou_thresh=0.3, max_frames_to_skip=5, max_trace_length=20, trackIdCount=0)

    # Variables initialization
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]

    count_vehicle = {'person': 0, 'motorbike': 0, 'car': 0, 'truck': 0, 'bicycle': 0, 'bus': 0}

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        if cap_from_stream:
            frame = cv2.resize(frame, (1280, 720))
        # frame = Image.fromarray(frame)

        # Detect and return centeroids of the objects in the frame
        result, centers, box_detected, obj_type = detector.detect(frame)
        result = frame

        print('Number of detections: ', len(centers))

        # If centroids are detected then track them
        if len(box_detected) > 0:

            # Track object using Kalman Filter
            tracker.Update(box_detected, obj_type)

            # For identified object tracks draw tracking line
            # Use various colors to indicate different track_id
            x1=0
            y1=0
            for i in range(len(tracker.tracks)):
                if len(tracker.tracks[i].trace) >= 5:
                    for j in range(len(tracker.tracks[i].trace) - 1):
                        # Draw trace line
                        x1 = tracker.tracks[i].trace[j][0][0]
                        y1 = tracker.tracks[i].trace[j][1][0]
                        x2 = tracker.tracks[i].trace[j + 1][0][0]
                        y2 = tracker.tracks[i].trace[j + 1][1][0]
                        clr = tracker.tracks[i].track_id % 9
                        cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)),
                                 track_colors[clr], 2)
                        
                    print(str(tracker.tracks[i].track_id))
                    cv2.putText(result,str(tracker.tracks[i].track_id),(int(x1),int(y1)),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255, 0, 255), 2, cv2.LINE_AA)

                    classes = tracker.tracks[i].get_obj()
                    
                    if (len(tracker.tracks[i].trace) >= 10) and (not tracker.tracks[i].counted):
                        bbox = tracker.tracks[i].ground_truth_box.reshape((4, 1))
                        tracker.tracks[i].counted = True
                        count_vehicle[classes] += 1
                        cv2.rectangle(result, (bbox[0][0], bbox[1][0]), (bbox[2][0], bbox[3][0]), color=(255, 0, 255),
                                      thickness=3)

        # Display the resulting tracking frame
        x = 30
        y = 30
        dy = 20
        i = 0
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL

        for key, value in count_vehicle.items():
            text = key + ':' + str(value)
            cv2.putText(result, text, (x, y + dy * i), font, 1, (255, 0, 255), 2, cv2.LINE_AA)
            i += 1
        cv2.rectangle(result, (ix, iy), (ex, ey), (0, 255, 0), 0)
        cv2.imshow('Tracking', result)
        out.write(result)

        # Check for key strokes
        k = cv2.waitKey(1) & 0xff
        if k == ord('n'):
            continue
        elif k == 27:  # 'esc' key has been pressed, exit program.
            break

    # When everything done, release the capture
    out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # execute main
    main()

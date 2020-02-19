import numpy as np
import cv2


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r], dtype=np.float32).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if w == 0.0:
        print('Stop: ', x)
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.], dtype=np.float32).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score], dtype=np.float32).reshape(
            (1, 5))


class KalmanFilter(object):
    def __init__(self, bbox):
        self.kalman = cv2.KalmanFilter(7, 4, 0)

        self.kalman.transitionMatrix = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
            dtype=np.float32)

        self.kalman.processNoiseCov = np.eye(7, dtype=np.float32)
        self.kalman.processNoiseCov[-1, -1] *= 0.01
        self.kalman.processNoiseCov[4:, 4:] *= 0.1

        self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32)
        self.kalman.measurementNoiseCov[2:, 2:] *= 10.

        self.kalman.errorCovPost = np.eye(7, dtype=np.float32)
        self.kalman.errorCovPost[4:, 4:] *= 100.  # give high uncertainty to the unobservable initial velocities
        self.kalman.errorCovPost *= 0.1

        self.kalman.statePost[:4] = convert_bbox_to_z(bbox)

        # lastResult saves the original box [x, y, s, r]
        self.lastResult = bbox

    def predict(self):
        prediction = self.kalman.predict()

        # lastResult saves the original box [x, y, s, r]
        self.lastResult = convert_x_to_bbox(prediction)
        return self.lastResult

    def correct(self, bbox, flag):
        if not flag:  # update using prediction
            measurement = self.lastResult
        else:  # update using detection
            measurement = bbox

        print('Flag: ', flag)
        print('bbox: ', bbox)
        measurement = convert_bbox_to_z(measurement.reshape((4, 1)))
        print('Measurement: ', measurement)
        estimate = self.kalman.correct(measurement)
        print('Estimate: ', estimate)

        # lastResult saves the original box [x, y, s, r]
        self.lastResult = convert_x_to_bbox(estimate)

        print('Last result: ', self.lastResult)
        return self.lastResult


img_height = 500
img_width = 500

x, y = 0, 0
bbox = [0, 0, 10, 10]
kf = KalmanFilter(bbox)
tracks = [[x / 2, y / 2]]

cnt = 0

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 15, (img_width, img_height))
while True:
    cnt += 1
    if cnt == 100:
        break
    print('Turn: ', cnt)
    kf.predict()
    dx = np.random.randint(0, min(10, img_height - 10 - x))
    dy = np.random.randint(0, min(10, img_width - 10 - y))

    measurement = np.array([[x + dx], [y + dy], [x + dx + 10], [y + dy + 10]], dtype=np.float32)
    estimate = kf.correct(measurement, 1)

    x += dx
    y += dy
    tracks.append([(estimate[0][0] + estimate[0][2]) / 2, (estimate[0][1] + estimate[0][3]) / 2])

    print('\n')

    u = int(estimate[0][0])
    v = int(estimate[0][1])
    img = np.zeros((img_height, img_width, 3), np.uint8)
    cv2.rectangle(img, (x, y), (x + 10, y + 10), color=(0, 0, 255))
    cv2.rectangle(img, (u, v), (u + 10, v + 10), color=(0, 255, 255))

    for i in range(len(tracks) - 1):
        cv2.line(img, (int(tracks[i][0]), int(tracks[i][1])), (int(tracks[i + 1][0]), int(tracks[i + 1][1])),
                 color=(255, 0, 0))

    cv2.imshow('image', img)
    out.write(img)
    code = cv2.waitKey(100)
    if code != -1:
        break

    if code in [27, ord('q'), ord('Q')]:
        break
    # cv2.waitKey(0)

out.release()

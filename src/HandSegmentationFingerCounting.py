import cv2
import numpy as np
from sklearn.metrics import pairwise

background = None
accumulated_weight = 0.5

# set up ROI
roi_top = 100
roi_bottom = 220
roi_right = 200
roi_left = 700


def calculate_accumulated_weight_avg(frame, accumulated_weight):
    global background

    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)


def segment(frame, threshold=20):
    global background

    diff = cv2.absdiff(background.astype("uint8"), frame)

    _, thresholded_hand_image = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    image, contours, hierarchy = cv2.findContours(thresholded_hand_image.copy(), cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    else:
        hand_segment = max(contours, key=cv2.contourArea)
        return (thresholded_hand_image, hand_segment)


def count_fingers(thresholded, hand_segment):
    conv_hull = cv2.convexHull(hand_segment)

    top = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    left = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    right = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])

    # calculate the center of the hand
    cX = (left[0] + right[0]) // 2
    cY = (top[1] + bottom[1]) // 2

    # calculate the Euclidean distance
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[left, right, top, bottom])[0]

    max_distance = distance.max()
    radius = int(0.9 * max_distance)
    circumference = (2 * np.pi * radius)

    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")

    cv2.circle(circular_roi, (cX, cY), radius, 255, 10)

    # extract the cut out on the thresholded hand image.
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # find contours
    image, contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    finger_count = 0

    for cnt in contours:

        (x, y, w, h) = cv2.boundingRect(cnt)

        out_of_wrist = ((cY + (cY * 0.25)) > (y + h))
        limit_points = ((circumference * 0.25) > cnt.shape[0])

        if out_of_wrist and limit_points:
            finger_count += 1

    return finger_count


# open webcam
cam = cv2.VideoCapture(0)

frame_count = 0

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()
    roi = frame[roi_top:roi_bottom, roi_right:roi_left]

    # apply grayscale and blur to ROI
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    if frame_count < 60:
        calculate_accumulated_weight_avg(gray, accumulated_weight)
        if frame_count <= 59:
            cv2.putText(frame_copy, "Getting background average...", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
            cv2.imshow("Finger count", frame_copy)

    else:
        hand = segment(gray)

        if hand is not None:
            thresholded, hand_segment = hand
            cv2.drawContours(frame_copy, [hand_segment + (roi_right, roi_top)], -1, (255, 0, 0), 1)
            fingers = count_fingers(thresholded, hand_segment)
            cv2.putText(frame_copy, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Thresholded", thresholded)

    cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0, 0, 255), 5)

    frame_count += 1

    cv2.imshow("Finger count", frame_copy)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()

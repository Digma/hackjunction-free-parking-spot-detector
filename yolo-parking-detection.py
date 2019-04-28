# Starting point: https://github.com/spmallick/learnopencv/blob/master/ObjectDetection-YOLO/object_detection_yolo.py
import cv2 as cv
import argparse
import numpy as np

from opencv.draw import draw_prediction
from opencv.io import load_open_cv_capture
from opencv.tracking import contain_similar_bounding_box
from yolo.yolo import setup_yolo_network, get_output_layer_names

"""
Constants
"""
CONFIDENCE_THRESH = 0.4  # Confidence threshold
NMS_THRESH = 0.4  # Non-maximum suppression threshold
INP_WIDTH = 416  # Width of network's input image
INP_HEIGHT = 416  # Height of network's input image

POSITION_HISTORY_SIZE = 20
IS_STATIC_THRESH = 0.7

MODEL_CONFIG = "darknet/cfg/yolov3.cfg" # model config (darknet default)
MODEL_WEIGHTS = "darknet/weights/yolov3.weights" # model weights
CLASS_LIST = "darknet/data/coco.names" # class names

CLASS_DICT = None
with open(CLASS_LIST, 'rt') as f:
    CLASS_DICT = f.read().rstrip('\n').split('\n')

"""
Arguments
"""
parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
parser.add_argument('--streaming', help='Path to video url')
parser.add_argument('--sampling', help='Sampling', type=int, default=24)
args = parser.parse_args()


def is_car(class_name):
    """
    :param class_name: string (as outputed by YOLO)
    :return: boolean
    """
    return class_name in str(["car", "truck", "bus"])


def is_parked_car(bounding_box, bounding_boxes_history, current_frame_idx, look_at_past_n, detection_ratio):
    """
    Determine if car is detected as parked
    
    :param bounding_box: Tuple of pixel positions (left, top, right, bottom)
    :param bounding_boxes_history: List of bounding box including frame number (frame_id, left, top, right, bottom)
    :param current_frame_idx: Index of current frame
    :param look_at_past_n: Number of frame to consider for classification
    :param detection_ratio: Minimum ratio to classify a car as parked
    :return: boolean
    """
    seen_in_previous = 0

    for n in range(1, look_at_past_n):
        bounding_boxes_time_n = [x for x in bounding_boxes_history if x[0] == current_frame_idx - n]

        if contain_similar_bounding_box(bounding_box, bounding_boxes_time_n):
            seen_in_previous += 1

    return seen_in_previous >= detection_ratio * look_at_past_n


def get_detected_objects(outs, confidence_threshold, frame_shape):
    """
    Scan through all the bounding boxes output from the network and keep only
    the ones with high confidence scores.

    Assign the box's class label as the class with the highest score.

    :param outs: row output weights produced by YOLO
    :param confidence_threshold: Confidence threshold (whether to keep object or not)
    :param frame_shape: (height, width in pixel of frame

    :return: class_ids, confidences, boxes (as arrays of size = number of object detected)
    """
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * frame_shape[1])
                center_y = int(detection[1] * frame_shape[0])
                width = int(detection[2] * frame_shape[1])
                height = int(detection[3] * frame_shape[0])
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_ids.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    return class_ids, confidences, boxes


def non_maximum_suppression(boxes, confidences, confidence_threshold, nms_threshold):
    """
    Remove the bounding boxes with low confidence using non-maxima suppression

    :param boxes: Array containing bounding boxes of detected objects
    :param confidences: Array of confidence values
    :param confidence_threshold: Confidence threshold (whether to keep object or not)
    :param nms_threshold: Non-maximum suppression threshold
    :return:
    """

    return cv.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)


def postprocess(frame, outs, car_positions_history, current_frame_num):

    car_positions = []

    class_ids, confidences, boxes = get_detected_objects(outs, CONFIDENCE_THRESH, frame.shape)
    object_ids = non_maximum_suppression(boxes, confidences, CONFIDENCE_THRESH, NMS_THRESH)

    for i in object_ids:
        i = i[0]
        id = class_ids[i]
        bounding_box = boxes[i]

        class_name = CLASS_DICT[id]

        # Only consider so-called cars
        if is_car(class_name):
            # If the car is detected as parked, color it blue
            if is_parked_car(bounding_box, car_positions_history,
                             current_frame_num, POSITION_HISTORY_SIZE,
                             IS_STATIC_THRESH):
                color = (255, 255, 0)
            else:
                color = (128, 128, 128)

            draw_prediction(frame, class_name, confidences[i], bounding_box, color)

            car_positions.append([current_frame_num, bounding_box])

    return car_positions_history + car_positions


def skip_next(cap, nframes):
    for i in range(nframes):
        ret = cap.grab()
        if not ret:
            cap.release()
            break

def main():
    # OpenCV init
    winName = 'Parking Detection'
    cv.namedWindow(winName, cv.WINDOW_NORMAL)
    cap, output_filename = load_open_cv_capture(args)

    # Get the video writer initialized to save the output video
    if (not args.image):
        fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
        vid_writer = cv.VideoWriter(output_filename, fourcc, 30,
                                    (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                                     round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    # YOLO init
    net = setup_yolo_network(MODEL_CONFIG, MODEL_WEIGHTS)

    # Variable init
    car_position_history = []
    current_frame_num = 0

    # Main loop
    while cv.waitKey(1) < 0:

        # get frame from the video
        has_frame, frame = cap.read()

        # Stop the program if reached end of video
        if not has_frame:
            print("Done processing !!!")
            print("Output file is stored as ", output_filename)
            cv.waitKey(3000)
            # Release device
            cap.release()
            break

        # Resize image and create a 4D blob from frame.
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (INP_WIDTH, INP_HEIGHT), [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_output_layer_names(net))

        # Remove the bounding boxes with low confidence
        car_position_history = postprocess(frame, outs, car_position_history, current_frame_num)

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # Write the frame with the detection boxes
        if args.image:
            cv.imwrite(output_filename, frame.astype(np.uint8))
        else:
            vid_writer.write(frame.astype(np.uint8))
            skip_next(cap, args.sampling-1)

        current_frame_num += 1

        cv.imshow(winName, frame)

    # cleanup the camera and close any open windows
    if not args.image:
        vid_writer.release()

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()

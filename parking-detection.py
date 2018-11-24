# Initial code based on: https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/
# https://github.com/spmallick/learnopencv/tree/master/ObjectDetection-YOLO

# %% Imports
import cv2 as cv
import argparse
import sys
import numpy as np
import os.path

# %% Initialize the parameters
confThreshold = 0.25  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 512       #Width of network's input image
inpHeight = 512      #Height of network's input image
frameSampling = 24
bbStaticNFrame=10 # Number of epoch to wait until considering car static

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
parser.add_argument('--mask', help='Path to black/white mask')
parser.add_argument('--skip', help='Skip the first X frames')
args = parser.parse_args()

# Load names of classes
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "darknet/cfg/yolov3.cfg"
modelWeights = "darknet/weights/yolov3.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def isCar(classId):
    return classes[classId] in str(["car", "truck", "bus"])

# %% Helper Functions

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom, color=(255, 178, 50)):

    if isCar(classId):
        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), color, 3)

        label = '%.2f' % conf

        # Get the label for the class name and its confidence
        if classes:
            assert(classId < len(classes))
            label = '%s:%s' % (classes[classId], label)

        #Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs, carLocationHistory, time, mask):
    lastCarLocations = []
    counterRecurrentCars = 0

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
            
        # Debug: Print center of bb
        cv.circle(frame, getCenterCoords(left, top, left + width, top + height), 5, (255,255,255))

        if isCar(classIds[i]):
            
            # For now ignoring size of box, only comparing center
            seenInPreviousFrameCounter = 0
            for i in range(1,bbStaticNFrame):
             lastFrameBBs = [x for x in carLocationHistory if x[0] == time-i]
             if (containSimilarBoundingBox(lastFrameBBs, left, top, left + width, top + height, width, mask)):
                seenInPreviousFrameCounter += 1

            if (seenInPreviousFrameCounter >= 0.7*bbStaticNFrame):
                counterRecurrentCars += 1
                drawPred(classIds[i], confidences[i], left, top, left + width, top + height, (255,255,0))
            else: 
                drawPred(classIds[i], confidences[i], left, top, left + width, top + height, (128,128,128))
            
            lastCarLocations.append([time, left, top, left + width, top + height ])
    
    addLabelToFrame(frame, "Parked Car Count: " + str(counterRecurrentCars))

    return carLocationHistory + lastCarLocations


# Font default text on frame
font                   = cv.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColor              = (255,255,0)
lineType               = 2

def addLabelToFrame(frame, text, bottomLeftCornerOfText=(30, 100)):    
    cv.putText(frame, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)  

def getCenterCoords(left, top, right, bottom):
    assert(right > left)
    assert(bottom > top)
    center_coord_y = int(left + (right - left) / 2.0)
    center_coord_x = int(top + (bottom - top) / 2.0)

    return (center_coord_x, center_coord_y)

# For now ignoring size of box, only comparing center
def containSimilarBoundingBox(boundingBoxes, left, top, right, bottom, width, mask, min_radius_thresh_px=25):
    # Handling empty case
    if len(boundingBoxes) <= 0:
        return False
    
    x, y = getCenterCoords(left, top, right, bottom)

    # If center of bounding box belong to mask, not a static car
    (mask_width, mask_height) = mask.shape

    if (x >= mask_width or y >= mask_height):
        return False

    if (mask[x,y] < 128):
        return False

    bb_centers = [ getCenterCoords(left, top, right, bottom) for (_, left, top, right, bottom) in boundingBoxes]
    
    minDistSquared = min([((x-a)**2 + (y-b)**2) for (a, b) in bb_centers])
    # Radius is larger for larger bounding boxes
    radius = max(min_radius_thresh_px, width/8.0 * (24.0 / frameSampling))

    return minDistSquared < radius**2

# Process inputs
winName = 'Street Parking Detection'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

# Load Input file
outputFile = "yolo_out_py.avi"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_yolo_out_py.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_yolo_out_py.avi'
else:
    # Webcam input
    cap = cv.VideoCapture(0)

# Load 1 frame to get size
ret, frame = cap.read()
height, width, channels = frame.shape

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

if(args.skip):
    # Skip some images at begigging of video
    for i in range(int(args.skip)):
        ret = cap.grab()
        if(not ret):
            cap.release()
            break

if(args.mask):
    # Load as grey-scale
    mask = cv.imread(args.mask, 0)
else:
    # Create a blank 300x300 black image
    mask = np.zeros((height, width), np.uint8)
    # Fill image with red color(set each pixel to red)
    mask[:] = 255


# Start with empty history
carLocationHistory = []
time = 0

while cv.waitKey(1) < 0:

    # get frame from the video
    hasFrame, frame = cap.read()

    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        # Release device
        cap.release()
        break

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    carLocationHistory = postprocess(frame, outs, carLocationHistory, time, mask)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Write the frame with the detection boxes
    if (args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8)) #cv.bitwise_and(frame,frame,mask = mask))
    else:
        vid_writer.write(frame.astype(np.uint8)) #cv.bitwise_and(frame,frame,mask = mask))
        # To stop duplicate images
        for i in range(frameSampling):
            # get frame from the video
            vid_writer.write(frame.astype(np.uint8))

            ret = cap.grab()
            if(not ret):
                cap.release()
                break


 
    cv.imshow(winName, frame) #cv.bitwise_and(frame,frame,mask = mask))

    time += 1

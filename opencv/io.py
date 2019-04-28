import cv2 as cv
import os
import pafy
import re
import sys


def file_does_not_exist(filepath):
    if not os.path.isfile(filepath):
        print("Input file ", filepath, " doesn't exist")
        sys.exit(1)


def load_open_cv_capture(args):
    if args.image:
        # Open the image file
        file_does_not_exist(args.image)

        cap = cv.VideoCapture(args.image)
        output_file = 'out/{}.{}'.format(args.image[:-4], 'jpg')

    elif args.video:
        # Open the video file
        file_does_not_exist(args.video)

        cap = cv.VideoCapture(args.video)
        output_file = 'out/{}.{}'.format(args.video[:-4], 'avi')
    elif args.streaming:
        url = args.streaming

        if "youtube" in url:
            video = pafy.new(args.streaming)
            name = video.title

            print("Launching video " + name)
            best = video.getbest()  # preftype="webm")
            url = best.url
            print("Url: " + url)
            print("Resolution: " + best.resolution)
            print("Extension selected: " + best.extension)

        else:
            # Only keep alphanumerical chars
            name = re.sub(r'\W+', '', url)

        cap = cv.VideoCapture(url)
        output_file = 'out/{}.{}'.format(name, 'avi')
    else:
        # Webcam input
        cap = cv.VideoCapture(0)
        output_file = 'out/{}.{}'.format('webcam', 'avi')

    print("Output will be saved as {}".format(output_file))

    return cap, output_file

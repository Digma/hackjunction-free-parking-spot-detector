#!/usr/bin/env bash

git clone https://github.com/pjreddie/darknet
cd darknet
make

mkdir -p weights
wget https://pjreddie.com/media/files/yolov3.weights -O ./weights/yolov3.weights

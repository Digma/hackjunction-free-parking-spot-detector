#!/usr/bin/env bash
cd darknet

for filename in ../$1/*; do
    ./darknet detect cfg/yolov3.cfg weights/yolov3.weights "$filename" -out "out_$(basename "$filename")"
done

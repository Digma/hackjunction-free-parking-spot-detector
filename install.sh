git clone https://github.com/sowson/darknet.git
cd darknet
make
cd ..

mkdir -p darknet/weights
wget https://pjreddie.com/media/files/yolov3.weights -O ./darknet/weights/yolov3.weights
wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true -O ./darknet/cfg/yolov3.cfg
wget https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true -O ./coco.names

# Create virtual env and install python dependencies
pip3 install virtualenv
pip3 install virtualenvwrapper

# if virtualenvwrapper.sh is in your PATH (i.e. installed with pip)
source `which virtualenvwrapper.sh`
mkvirtualenv parkmate
workon parkmate
pip3 install -r requirements.txt 

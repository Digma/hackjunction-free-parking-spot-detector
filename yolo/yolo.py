import cv2 as cv


def setup_yolo_network(model_config, model_weights):
    net = cv.dnn.readNetFromDarknet(model_config, model_weights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    return net


# Get the names of the output layers
def get_output_layer_names(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

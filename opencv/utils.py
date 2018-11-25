import cv2 as cv

# Font default text on frame
font                   = cv.FONT_HERSHEY_DUPLEX
fontScale              = 1.2
lineType               = 2

def addLabelToFrame(frame, text, bottomLeftCornerOfText=(30, 100), fontColor = (255,255,0)):    
    cv.putText(frame, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)  

def getCenterCoords(left, top, right, bottom):
    assert(right > left)
    assert(bottom > top)
    center_coord_y = int(left + (right - left) / 2.0)
    center_coord_x = int(top + (bottom - top) / 2.0)

    return (center_coord_x, center_coord_y)

# For now ignoring size of box, only comparing center
def containSimilarBoundingBox(boundingBoxes, left, top, right, bottom, width, mask, min_radius_thresh_px=25, frameSampling=24):
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
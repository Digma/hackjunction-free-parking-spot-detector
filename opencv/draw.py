import cv2 as cv


# Draw the predicted bounding box
def draw_prediction(frame, class_name, conf, bounding_box, color=(255, 178, 50)):
    left, top, width, height = bounding_box

    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (left+width, top+height), color, 3)

    # Draw label at top of bounding box
    label = '%s:%.2f' % (class_name, conf)
    label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.3, 1)

    top = max(top, label_size[1])
    cv.rectangle(frame, (left, top - round(1.1 * label_size[1])), (left + round(1.1 * label_size[0]), top + base_line), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

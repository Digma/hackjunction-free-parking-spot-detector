def get_center_coords(bounding_box):
    left, top, width, height = bounding_box

    center_coord_y = int(left + width / 2.)
    center_coord_x = int(top + height / 2.)

    return center_coord_x, center_coord_y


def get_shortest_side_lenght(bounding_box):
    left, top, width, height = bounding_box

    return min(width, height)


def contain_similar_bounding_box(bounding_box, bounding_boxes_in_frame,
                                 min_radius_thresh_px=20):
    """
    :param bounding_box: Tuple of pixel positions (left, top, right, bottom)
    :param bounding_boxes_in_frame: List of all bounding boxes at a given frame
    :param min_radius_thresh_px: Minimum distance to match bounding box (usefule for small objects)
    :return: boolean if at least 1 bounding box is matching
    """

    # Handling empty case
    if len(bounding_boxes_in_frame) <= 0:
        return False

    # Compute center of current bounding box
    x, y = get_center_coords(bounding_box)

    # Get center of previous bounding boxes
    bb_centers = [get_center_coords(bounding_box) for (_, bounding_box) in bounding_boxes_in_frame]
    min_dist_squared = min([((x - a) ** 2 + (y - b) ** 2) for (a, b) in bb_centers])

    # Increase radius for larger bounding boxes
    shortest_side = get_shortest_side_lenght(bounding_box)
    radius = max(min_radius_thresh_px, shortest_side / 8.0)

    return min_dist_squared < radius ** 2

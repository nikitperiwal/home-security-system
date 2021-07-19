import cv2
import numpy as np
from utils.check_param import verify_border_args


def create_borders(img, coords, labels) -> np.ndarray:
    """
    Creating the borders for a single frame from the coordinates given
    """

    for label, coord in zip(labels, coords):
        # Setting color as red to indicate person is unregistered
        color = (230, 0, 0) if label == "Cannot Identify" else (0, 0, 230)
        x1, y1, x2, y2 = coord
        # For bounding box
        img = cv2.rectangle(img, (x1, y1), (x1+x2, y1+y2), color, 2)

        # For the text background
        # Finds space required by the text so that we can put a background with that amount of width.
        (w, h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # Prints the text.
        img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)

        # For printing text
        img = cv2.putText(img, label, (x1, y1),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def add_borders(frame_list: np.ndarray, frame_coords: list, face_labels: list) -> list:
    """
    Returns a list of frames with bounding boxes

    Parameters
    ----------
    frame_list: a list of frames
    frame_coords: coordinates of the face location
    face_labels: a list containing the labels for faces detected

    Returns
    -------
    list_with_bb: A list of frames containing the bounding boxes
    """

    verify_border_args(frame_list, frame_coords, face_labels)
    list_with_bb = []
    for frame, coords, labels in zip(frame_list, frame_coords, face_labels):
        if labels:
            list_with_bb.append(create_borders(frame, coords, labels))
        else:
            list_with_bb.append(frame)
    return list_with_bb

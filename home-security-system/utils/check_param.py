import cv2
import numpy as np
from queue import Queue


def verify_motion_args(vid_stream, vid_index, q, threshold_val, min_contour_area, frame_padding, update_init_thres):
    """
    Verify the arguments received from motion_detection.motion_detection
    """
    if not isinstance(vid_stream, cv2.VideoCapture):
        raise TypeError("vid_stream must be a VideoCapture object")
    if not isinstance(vid_index, int):
        raise TypeError("The vid_index must be an int")
    if not isinstance(q, Queue):
        raise TypeError("The queue argument should be of type queue.Queue")
    if not isinstance(threshold_val, int) or threshold_val > 255:
        raise Exception("threshold_val should be an integer between 0 and 255")
    if not isinstance(min_contour_area, int):
        raise TypeError("min_contour_area should be an integer")
    if not isinstance(update_init_thres, int):
        raise TypeError("update_init_thres should be an integer")
    if not isinstance(frame_padding, int):
        raise TypeError("frame_padding should be an integer")


def verify_stream_args(q, video_cap):
    """
    Verify the arguments received from video_stream.video_stream
    """
    if not isinstance(video_cap, cv2.VideoCapture) and video_cap is not None:
        raise ValueError("video_cap must be a VideoCapture object or None")
    if not isinstance(q, Queue):
        raise TypeError("Input queue is not of type queue.Queue")


def verify_save_args(q, time_q,  fps):
    """
    Verify the arguments received from video_stream.save_queue
    """
    if not isinstance(q, Queue):
        raise TypeError("Input queue is not of type queue.Queue")
    if not isinstance(time_q, Queue):
        raise TypeError("Time queue is not of type queue.Queue")
    if not isinstance(fps, int):
        raise TypeError("FPS is not of type int")


def verify_border_args(frame_list, frame_coords, face_labels):
    """
    Verifies the arguments passed to add_borders 
    """
    if not isinstance(frame_list, np.ndarray):
        raise TypeError("frame_list should be of type: np.ndarray")
    if not isinstance(frame_coords, list):
        raise TypeError("frame_coords should be of type: list")
    if not isinstance(face_labels, list):
        raise TypeError("face_labels should be of type: list")


def verify_hss_args(vid_streams):
    """
    Verifies the arguments passed to the constructor of HomeSecuritySystem
    """
    if vid_streams is None:
        print("\033[93mWarning: There's no video stream passed, assigning a camera by default \033[00m")
        return
    if not isinstance(vid_streams, list) and not isinstance(vid_streams, tuple) and vid_streams is not None:
        raise TypeError("vid_streams must be a tuple containing VideoCapture object")
    for vid_stream in vid_streams:
        if not isinstance(vid_stream, cv2.VideoCapture):
            raise ValueError("The contents of video_streams must be a VideoCapture object")


def verify_stream(vid_stream, vid_src):
    """
    Checks if the source of VideoCapture object is a camera
    """
    if isinstance(vid_src, str):
        print("\033[93mWarning: The video is streaming from a file than a camera \033[00m")
        return
    if not isinstance(vid_stream, cv2.VideoCapture) and vid_stream is not None:
        raise ValueError("The contents of video_streams must be a VideoCapture object")
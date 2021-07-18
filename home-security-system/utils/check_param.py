import cv2
from queue import Queue


def verify_motion_args(q, processed_q, time_q, threshold_val, min_contour_area, update_init_thres, frame_padding):
    """
    Verify the arguments received from motion_detection.motion_detection
    """
    if not isinstance(q, Queue):
        raise TypeError("Input queue is not of type queue.Queue")
    elif not isinstance(processed_q, Queue):
        raise TypeError("Processed queue is not of type queue.Queue")
    elif not isinstance(time_q, Queue):
        raise TypeError("Time queue is not of type queue.Queue")
    elif not isinstance(threshold_val, int) or threshold_val > 255:
        raise Exception("threshold_val should be an integer between 0 and 255")
    elif not isinstance(min_contour_area, int):
        raise TypeError("min_contour_area should be an integer")
    elif not isinstance(update_init_thres, int):
        raise TypeError("update_init_thres should be an integer")
    elif not isinstance(frame_padding, int):
        raise TypeError("frame_padding should be an integer")


def verify_stream_args(q, video_cap):
    """
    Verify the arguments received from video_stream.video_stream
    """
    if (not isinstance(video_cap, tuple) and video_cap is not None) or \
            (not isinstance(video_cap[0], cv2.VideoCapture) or not isinstance(video_cap[1], bool)):
        raise ValueError("video_cap must be a tuple containing VideoCapture and cam_flag or None")
    elif not isinstance(q, Queue):
        raise TypeError("Input queue is not of type queue.Queue")


def verify_save_args(q, time_q,  fps):
    """
    Verify the arguments received from video_stream.save_queue
    """
    if not isinstance(q, Queue):
        raise TypeError("Input queue is not of type queue.Queue")
    elif not isinstance(time_q, Queue):
        raise TypeError("Time queue is not of type queue.Queue")
    elif not isinstance(fps, int):
        raise TypeError("FPS is not of type int")


def verify_face_register(name, face_images, registered):
    """"
    Verifies the Register_Faces parameter.
    """
    if not isinstance(name, str):
        raise ValueError("name should have d-type: str")
    if len(name) == 0:
        raise ValueError("name cannot be empty")
    if name in list(registered):
        raise ValueError("Name already exists in Secure Faces List\nPlease enter another Name")
    if face_images > 3:
        print("Only first 3 images for the person would be registered")

import cv2
import numpy as np


def record_n_frames(n: int, cap) -> list:
    """
    Records n frames and returns the numpy array of frames

    Parameters
    ----------
    n: Number of frames

    Returns
    --------
    frames: A numpy array of recorded frames
    """
    frames = []
    for i in range(10):
        ret, frame = cap.read()
    for i in range(n):
        ret, frame = cap.read()

        frames.append(cv2.resize(frame, (128, 128)))
    return np.array(frames)

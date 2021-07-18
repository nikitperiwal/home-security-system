import cv2
import numpy as np


cascade_model = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")


def remove_extra_faces(detected_list):
    """
    Removes same faces in from sequence of frames and returns face_list, face_index.

    Parameters
    -----------
    detected_list: list containing co-ordinates of the faces in the frame

    Returns
    --------
    face_list: list containing co-ordinates of the faces in the frame
    face_index: list of face_index that points to faces in face_list
    """

    prev_length = 0
    face_list = []
    face_index = []

    for detected in detected_list:
        if len(detected) == 0:
            prev_length = 0
            face_index.append([])

        elif len(detected) == prev_length:
            index = range(len(face_list) - len(detected), len(face_list))
            index = list(index)
            face_index.append(index)

        else:
            face_list.extend(detected)
            index = range(len(face_list) - len(detected), len(face_list))
            index = list(index)
            face_index.append(index)
            prev_length = len(detected)

    return np.array(face_list), face_index


def extract_faces(frame_list, face_index, coords_list):
    """
    Extract faces from the images using a list of co-ordinates passed.

    Parameters
    -----------
    frame_list: list of frames from a video
    face_index: list of face_index that points to faces in face_list
    coords_list: list containing co-ordinates of the faces in the frame

    Returns
    --------
    faces: list containing extracted faces from the image
    """

    def fix_coords(face_list, resolution=(1280, 720), delta=30):
        new_list = []
        for x1, y1, w, h in face_list:
            x1 = max(0, x1 - delta)
            y1 = max(0, y1 - delta)
            x2 = min(resolution[0], x1 + max(128, w + delta))
            y2 = min(resolution[1], y1 + max(128, h + delta))
            new_list.append((x1, y1, x2, y2))
        return new_list

    coords_list = fix_coords(coords_list)

    prev_index = []
    faces = []
    for frame, indexes in zip(frame_list, face_index):
        if len(indexes) > 0 and indexes != prev_index:
            prev_index = indexes
            for index in indexes:
                (x1, y1, x2, y2) = coords_list[index]
                face = frame[y1:y2, x1:x2]
                faces.append(cv2.resize(face, (128, 128)))

    return np.array(faces)


def detect_from_image(image: np.ndarray):
    """
    Detects faces from a frame in video and returns a list of all face images.

    Parameters
    -----------
    image: image of video frame

    Returns
    --------
    faces: list containing co-ordinates of the faces in the frame
    """

    global cascade_model

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade_model.detectMultiScale(
        gray,
        scaleFactor=1.4,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return np.array(faces)


def detect_from_video(frame_list):
    """
    Detects faces from a list of video frames and returns face_list, face_index.

    Parameters
    -----------
    frame_list: list of frames from a video

    Returns
    --------
    coord_list: list cont
    face_list: list containing co-ordinates of the faces in the frame
    face_index: list of face_index that points to faces in face_list
    """

    frame_coords = []
    for frame in frame_list:
        frame_coords.append(detect_from_image(frame))

    face_coords, face_index = remove_extra_faces(frame_coords)
    detected_faces = extract_faces(frame_list=frame_list, face_index=face_index, coords_list=face_coords)
    return frame_coords, face_index, detected_faces


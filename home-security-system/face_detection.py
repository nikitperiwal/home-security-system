import cv2
import numpy as np


cascade_model = cv2.CascadeClassifier("data/models/haarcascade_frontalface_default.xml")


def fix_coords(coords, resolution=(1280, 720), delta=50):
    """ Increases the size of the coordinates """

    x1, y1, w, h = coords
    x1 = max(0, x1 - delta)
    y1 = max(0, y1 - delta)
    x2 = min(resolution[0], x1 + max(128, w + delta))
    y2 = min(resolution[1], y1 + max(128, h + delta))
    return x1, y1, x2, y2


def is_blurred(img, blur_threshold=220):
    """ Returns blur_score if image is blurred, otherwise False """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.convertScaleAbs(cv2.Laplacian(gray, 3))
    score = np.max(score)
    if score < blur_threshold:
        return score
    return False


def extract_face(frame, coords):
    """ Extracts the face from the coords; Returns NONE if face blurred"""

    (x1, y1, x2, y2) = fix_coords(coords)
    face = cv2.resize(frame[y1:y2, x1:x2], (128, 128))
    return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)


def remove_extra_faces(frame_list, coords_list):
    """
    Removes same faces in from sequence of frames and returns face_list, face_index.

    Parameters
    -----------
    frame_list: list containing all the frames
    coords_list: list containing co-ordinates of the faces in the frame

    Returns
    --------
    face_list: list containing the faces in the frame
    face_index: list of face_index that points to faces in face_list
    """

    prev_length = 0
    face_list = []
    face_index = []
    blur_list = []

    for i in range(len(coords_list)):
        if len(coords_list[i]) == 0:
            prev_length = 0
            blur_list = []
            face_index.append([])

        elif len(coords_list[i]) == prev_length:
            # Appends to face_index
            index = range(len(face_list) - len(coords_list[i]), len(face_list))
            index = list(index)
            face_index.append(index)

            # Continues to next frame if no faces blurred.
            if not any(blur_list):
                continue

            # If faces blurred, replace with new faces.
            for j, coords in enumerate(coords_list[i]):
                if blur_list[j]:
                    face = extract_face(frame_list[i], coords)
                    blur_score = is_blurred(face)
                    if blur_score > blur_list[j]:
                        blur_list[j] = blur_score
                        face_list[index[j]] = face

        else:
            # Appends face to face_list and a blur flags to blur list.
            blur_list = []
            for coords in coords_list[i]:
                face = extract_face(frame_list[i], coords)
                blur_list.append(is_blurred(face))
                face_list.append(face)

            # Appends to face_index
            index = range(len(face_list) - len(coords_list[i]), len(face_list))
            index = list(index)
            face_index.append(index)

            prev_length = len(coords_list[i])

    return np.array(face_list), face_index


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

    # Increasing Image Contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Getting the face coordinates
    faces = cascade_model.detectMultiScale(
        gray,
        scaleFactor=1.4,
        minNeighbors=5,
        minSize=(32, 32)
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

    detected_faces, face_index = remove_extra_faces(frame_list, frame_coords)
    return frame_coords, face_index, detected_faces

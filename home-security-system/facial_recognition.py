import os
import cv2
import numpy as np

from utils.add_borders import add_borders
from utils.alert import create_notification
from utils.remove_file import remove_file

from face_detection import detect_from_video
from face_recognition import encode_images, check_similarity


resolution = (1280, 720)


def read_video(filepath):
    """ Reads the video from filename and returns it, along with video fps """

    vid_stream = cv2.VideoCapture(filepath)

    frame_count = int(vid_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid_stream.get(cv2.CAP_PROP_FPS))

    frames = np.empty((frame_count, resolution[1], resolution[0], 3), np.dtype('uint8'))
    for i in range(frame_count):
        ret, frame = vid_stream.read()
        frame = cv2.resize(frame, resolution)
        frames[i] = frame

    vid_stream.release()
    return frames, fps


def save_video(frame_list, filename, fps):
    """ Writes the video to the filename specified """
    video_path = f"Detected Videos/{filename}"
    directory = os.path.dirname(video_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, resolution)
    for frame in frame_list:
        out.write(frame)
    out.release()


def raise_alert(filename, face_labels, intruder_threshold):
    """ Raises a Native OS Alert if intruder detected """

    intruder_count = sum([1 if "Cannot Identify" in label else 0 for label in face_labels])
    intruder_count /= len(face_labels)

    if intruder_count >= intruder_threshold:
        message = f"Intruder detected! Please check {filename}!"
        create_notification("Home Security System", message)


def check_registered(registered_faces, detected_faces):
    """
    Compares the passed detected_faces to registered_faces and return the labels.

    Parameters
    -----------
    registered_faces: dict of registered faces in HSS.
    detected_faces: list of faces detected from video

    Returns
    --------
    labels: labels of faces passed
    """

    num_reg = len(registered_faces)
    names = list(registered_faces.keys())
    num_det = len(detected_faces)

    if num_reg == 0:
        return ["Cannot Identify"] * num_det

    labels = []
    for d in detected_faces:
        predictions = []
        for name in names:
            t1 = registered_faces[name]
            t2 = np.tile(d, (len(t1), 1))

            pred = np.mean(check_similarity(t1, t2))
            predictions.append(pred)
        predictions = np.array(predictions)

        def get_labels(arr):
            index = arr.argmin()
            if arr[index] <= 0.5:
                return names[index]
            return "Cannot Identify"

        labels.append(get_labels(predictions))
    return labels


def facial_recognition(registered_faces, filepath, intruder_threshold=0.2):
    """"
    Runs Facial Recognition on passed video file and saves it with bounding boxes.
    Raises an alert if intruder found.

    Parameters
    -----------
    registered_faces: Dict containing people already registered
    filepath: filename for the video to run facial recognition on
    intruder_threshold: threshold upon which alert generated
    """

    try:
        frame_list, fps = read_video(filepath)
        filename = filepath[-25:]
        # Run face detection and label detected faces
        frame_coords, face_index, detected_faces = detect_from_video(frame_list)

        # Runs Face Recognition, if faces are detected.
        if len(detected_faces) > 0:
            # Getting labels for detected face.
            detected_faces = encode_images(detected_faces)
            detected_labels = check_registered(registered_faces, detected_faces)

            # Tag labels to individual frames
            face_labels = []
            for indexes in face_index:
                face_labels.append([detected_labels[index] for index in indexes])
            # adding borders to frames where face was detected
            frame_list = add_borders(frame_list, frame_coords, face_labels)
            # Raising alert
            raise_alert(filename, face_labels, intruder_threshold)

        save_video(frame_list, filename, fps)
        remove_file(filepath)

    except Exception as e:
       print(f"Error while facial recognition on file: {filepath}\n Error: {e}")

    finally:
        pass


def start_facial_recognition(registered_faces, file_queue):
    """ Continously checks the file_queue for files. If file found, runs facial recognition """

    filepath = None
    try:
        while True:
            filepath = file_queue.get()

            if filepath == "EXIT":
                print("Exiting facial recognition")
                break
            facial_recognition(registered_faces, filepath)

    except KeyboardInterrupt:
        if not filepath:
            filepath = file_queue.get()
        while True:
            if filepath == "EXIT":
                print("Exiting facial recognition")
                break
            facial_recognition(registered_faces, filepath)
            filepath = file_queue.get()

    except Exception as e:
        print(f"Exception occured when starting facial recognition \nError: {e}")

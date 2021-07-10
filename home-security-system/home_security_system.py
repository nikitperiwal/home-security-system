import cv2
import numpy as np
from queue import Queue
from threading import Thread

from face_detection import detect_from_video
from face_recognition import convolve_images, check_similarity
from motion_detection import motion_detection
from video_stream import video_stream, save_queue


class HomeSecuritySystem:
    def __init__(self):
        self.registered_faces = {}

    def face_register(self, name: str, face_images: np.ndarray):
        """
        Registers faces as secure, along with name.

        Parameters
        -----------
        name       : Name of the person to register
        face_images: Numpy array of the face images for the person
        """

        if not isinstance(name, str):
            raise ValueError("name should have d-type: str")
        if len(name) == 0:
            raise ValueError("name cannot be empty")
        if name in self.registered_faces.keys():
            raise ValueError("Name already exists in Secure Faces List\nPlease enter another Name")

        if len(face_images) > 3:
            print("Only first 3 images for the person would be registered")
            face_images = face_images[:3]

        self.registered_faces[name] = convolve_images(face_images)

    def check_registered(self, detected_faces):
        detected_faces = convolve_images(detected_faces)

        num_det = len(detected_faces)
        num_reg = len(self.registered_faces)
        names = list(self.registered_faces.keys())

        tensor1 = []
        tensor2 = []
        for d in detected_faces:
            for name in names:
                tensor1.extend([d] * len(self.registered_faces[name]))
                tensor2.extend(self.registered_faces[name])
        tensor1 = np.array(tensor1)
        tensor2 = np.array(tensor2)

        pred = check_similarity(tensor1, tensor2)
        pred = pred.reshape(num_det, num_reg, -1)
        pred = pred.mean(axis=-1)

        def get_labels(arr):
            index = arr.argmax()
            if arr[index] >= 0.5:
                return names[index]
            return None

        labels = [get_labels(p) for p in pred]
        return labels

    @staticmethod
    def remove_extra_faces(detected_list):
        prev_length = 0
        new_list = []

        def check_similar(clist1, clist2, threshold=20):
            for c1, c2 in zip(clist1, clist2):
                x = (c2[0] - c1[0]) ** 2
                y = (c2[1] - c1[1]) ** 2
                if (x+y)*0.5 > threshold:
                    return True
            return False

        for detected in detected_list:
            if len(detected) == prev_length:
                if check_similar(detected, new_list[-1]):
                    new_list.append(detected)
                    prev_length = len(detected)
                else:
                    pass
            else:
                new_list.append(detected)
                prev_length = len(detected)

        return new_list

    # TODO
    @staticmethod
    def start_detecting(vid_stream: tuple = None, video_src: str = None):
        """
        Starts detecting motion and recognises a facees of people

        Parameters
        ----------
        vid_stream: A tuple containing cv2.VideoCapture object and a cam_flag
        video_src: Selects the source for cv2.VideoCapture object
        """
        if vid_stream is None:
            vid_stream = (cv2.VideoCapture(0), True) if video_src is None or video_src == "" \
                else (cv2.VideoCapture(video_src), False)

        queue, processed_queue, time_queue = Queue(), Queue(), Queue()
        t1 = Thread(target=video_stream, args=(queue, vid_stream))
        stream_fps = int(vid_stream[0].get(cv2.CAP_PROP_FPS))
        t2 = Thread(target=motion_detection, args=(queue, processed_queue, time_queue, stream_fps))
        t3 = Thread(target=save_queue, args=(processed_queue, time_queue, stream_fps))

        t1.start()
        t2.start()
        t3.start()

        t1.join()
        t2.join()
        t3.join()

        # get queue of frames
        # pass queue to detect_faces()
        # get return coordinates list
        # pass co-ordinates list to remove_extra_faces

if __name__ == '__main__':
    HomeSecuritySystem().start_detecting((cv2.VideoCapture(0), True))

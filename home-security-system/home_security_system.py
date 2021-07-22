import cv2
import numpy as np

from threading import Thread
from multiprocessing import Process, Queue

from face_recognition import encode_images
from motion_detection import motion_detection
from facial_recognition import start_facial_recognition

from utils.register_face import *


class HomeSecuritySystem:
    def __init__(self):
        self.vid_streams = []
        self.stream_status = []
        self.motion_threads = []

        self.registered_faces = load_faces()
        self.motion_files = Queue()
        self.surveillance_process = Process(target=start_facial_recognition,
                                            args=(self.registered_faces, self.motion_files))

    def add_video_stream(self, vid_stream):
        """
        Adds a video_stream to the security system.

        Parameters
        -----------
        vid_stream: The video stream object, can be cv2.VideoCapture obj, filename or IP address.
        """
        if not isinstance(vid_stream, cv2.VideoCapture):
            vid_stream = cv2.VideoCapture(vid_stream)
        if vid_stream is None or not vid_stream.isOpened():
            return "Please send a correct Video Stream format."

        # Appends to vid_stream and returns the index.
        self.vid_streams.append(vid_stream)
        self.stream_status.append(False)
        return len(self.vid_streams) - 1

    def kill_video_stream(self, index):
        """ Kills the video_stream at the given index """

        if index >= len(self.vid_streams):
            print(f"Camera {index} not found")
            return
        self.vid_streams.pop(index).release()
        self.stream_status.pop(index)

    def register_person(self, name: str, face_images: np.ndarray):
        """
        Registers a Person, along with name and images.

        Parameters
        -----------
        name       : Name of the person to register
        face_images: Numpy array of the face images for the person
        """
        try:
            verify_register_person(name, len(face_images), self.registered_faces.keys())
            self.registered_faces[name] = encode_images(face_images[:3])
            save_faces(self.registered_faces)
        except Exception as e:
            print(e)

    def delete_person(self, name: str):
        """
        Deletes the registered person from the database.

        Parameters
        -----------
        name : Name of the person to delete
        """

        verify_delete_person(name, self.registered_faces.keys())
        self.registered_faces.pop(name, None)
        save_faces(self.registered_faces)

    def rename_person(self, old_name: str, new_name: str):
        """
        Renames the person already existing in the database.

        Parameters
        -----------
        old_name : Name of the person to change.
        new_name : New name of the person.
        """

        verify_rename_person(old_name, new_name, self.registered_faces.keys())

        self.registered_faces[new_name] = self.registered_faces[old_name]
        self.registered_faces.pop(old_name, None)
        save_faces(self.registered_faces)

    def start_camera(self, cam_index: int):
        """
        Starts detecting for any potential dangerous situation

        Parameters
        ----------
        cam_index: Specifies the camera
        """

        try:
            if not isinstance(cam_index, int):
                raise TypeError("Camera index should be an integer")
            elif cam_index < 0 or cam_index >= len(self.vid_streams):
                raise ValueError(f"Cam Index out of range. Camera at index {cam_index} could not be found.")

            if self.stream_status[cam_index]:
                return "Camera already running"

            vid_stream = self.vid_streams[cam_index]

            # Creating the motion_detection thread and storing it
            motion_thread = Thread(target=motion_detection,
                                   args=(vid_stream, cam_index, self.motion_files))
            motion_thread.start()
            self.surveillance_process.start()
            self.motion_threads.append(motion_thread)
            self.stream_status[cam_index] = True

            while motion_thread.is_alive():
                pass
        
        except KeyboardInterrupt:
            print("Please wait till the termination is completed to avoid corrupt save files")

        except Exception as e:
            print(e)

        finally:
            self.kill_video_stream(cam_index)
            motion_thread.join()
            self.surveillance_process.join()
            print("Program terminated")


if __name__ == '__main__':
    # my_image = cv2.cvtColor(cv2.resize(cv2.imread("IGNORE/my_image.jpg"), (128, 128)), cv2.COLOR_BGR2RGB)
    # my_image1 = cv2.cvtColor(cv2.resize(cv2.imread("IGNORE/my_image1.jpg"), (128, 128)), cv2.COLOR_BGR2RGB)
    # my_image2 = cv2.cvtColor(cv2.resize(cv2.imread("IGNORE/my_image2.jpg"), (128, 128)), cv2.COLOR_BGR2RGB)
    # my_image = np.array([my_image, my_image1, my_image2])
    # server = HomeSecuritySystem()
    # server.register_person("Nikit", my_image)
    # server.register_person("Niranjan", my_image)

    cam_0 = cv2.VideoCapture(0)
    server = HomeSecuritySystem()
    vid_index = server.add_video_stream(vid_stream=cam_0)
    server.start_camera(vid_index)

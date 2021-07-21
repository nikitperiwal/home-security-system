import cv2
import numpy as np
from queue import Queue
from threading import Thread

from face_recognition import encode_images
from motion_detection import motion_detection
from video_stream import video_stream, save_queue

from utils.register_face import load_faces, save_faces
from utils.check_param import verify_face_register


class HomeSecuritySystem:
    def __init__(self):
        self.vid_streams = []
        self.registered_faces = load_faces()

    def add_video_stream(self, vid_stream):
        """
        Adds a video_stream to the security system.

        Parameters
        -----------
        vid_stream: The video stream object, can be cv2.VideoCapture obj, filename or IP address.
        """

        # TODO - CHECK ONE
        if not isinstance(vid_stream, cv2.VideoCapture):
            vid_stream = cv2.VideoCapture(vid_stream)
        if vid_stream is None or not vid_stream.isOpened():
            return "Please send a correct Video Stream format."

        # Appends to vid_stream and returns the index.
        self.vid_streams.append(vid_stream)
        return len(self.vid_streams)

    def face_register(self, name: str, face_images: np.ndarray):
        """
        Registers faces as secure, along with name.

        Parameters
        -----------
        name       : Name of the person to register
        face_images: Numpy array of the face images for the person
        """

        verify_face_register(name, len(face_images), self.registered_faces.keys())
        self.registered_faces[name] = encode_images(face_images[:3])
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

            self.start_detecting(vid_stream=self.vid_streams[cam_index])
        except Exception as e:
            print(e)

    def start_detecting(self, vid_stream: cv2.VideoCapture = None, vid_src: int = None):
        """
        Starts detecting motion and recognises a faces of people

        Parameters
        ----------
        vid_stream: A cv2.VideoCapture object
        vid_src: Selects the source for cv2.VideoCapture object
        """

        queue, processed_queue, time_queue, final_queue = Queue(), Queue(), Queue(), Queue()

        stream_fps = int(vid_stream.get(cv2.CAP_PROP_FPS))

        t1 = Thread(target=video_stream, args=(queue, vid_stream), daemon=True)
        t2 = Thread(target=motion_detection, args=(queue, processed_queue, time_queue, stream_fps), daemon=True)

        # TODO bug fix fps (may vary from pc)
        t3 = Thread(target=self.facial_recognition, args=(processed_queue, final_queue))
        t4 = Thread(target=save_queue, args=(final_queue, time_queue, stream_fps // 3 + 1))

        try:
            t1.start()
            t2.start()
            t3.start()
            t4.start()
            while t3.is_alive():
                pass

        except KeyboardInterrupt:
            print("Stopping the recording")

        finally:
            vid_stream.release()
            t3.join()
            t4.join()


if __name__ == '__main__':
    # my_image = cv2.cvtColor(cv2.resize(cv2.imread("IGNORE/my_image.jpg"), (128, 128)), cv2.COLOR_BGR2RGB)
    # my_image1 = cv2.cvtColor(cv2.resize(cv2.imread("IGNORE/my_image1.jpg"), (128, 128)), cv2.COLOR_BGR2RGB)
    # my_image2 = cv2.cvtColor(cv2.resize(cv2.imread("IGNORE/my_image2.jpg"), (128, 128)), cv2.COLOR_BGR2RGB)
    # my_image = np.array([my_image, my_image1, my_image2])

    # server = HomeSecuritySystem()
    # server.face_register("Nikit", my_image)

    # server.face_register("Niranjan", my_image)

    cam_0 = cv2.VideoCapture("IGNORE/video.mp4")
    server = HomeSecuritySystem()
    vid_index = server.add_video_stream(vid_stream=cam_0)
    server.start_camera(vid_index)

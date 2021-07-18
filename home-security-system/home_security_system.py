import cv2
import numpy as np
from queue import Queue
from threading import Thread

from face_detection import detect_from_video
from face_recognition import encode_images, check_similarity
from video_stream import video_stream, save_queue
from motion_detection import motion_detection
from utils.record_n_frames import record_n_frames
from utils.add_borders import add_borders


class HomeSecuritySystem:
    def __init__(self, register:bool = False):
        #  TODO save registered faces and remove register flag
        self.registered_faces = {}
        self.register_flag = register

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
        self.registered_faces[name] = encode_images(face_images)

    # def check_if_already_registered(self):
    # Pass

    def check_registered(self, detected_faces):
        """
        Compares the passed detected_faces to registered_faces and return the labels.

        Parameters
        -----------
        detected_faces: list of faces detected from video

        Returns
        --------
        labels: labels of faces passed
        """

        num_det = len(detected_faces)
        num_reg = len(self.registered_faces)
        names = list(self.registered_faces.keys())
        tensor1 = []
        tensor2 = []
        for d in detected_faces:
            # TODO when names are empty
            for name in names:
                tensor1.extend([d] * len(self.registered_faces[name]))
                tensor2.extend(self.registered_faces[name])
        tensor1 = np.array(tensor1)
        tensor2 = np.array(tensor2)
        pred = check_similarity(tensor1, tensor2)
        pred = pred.reshape(num_det, num_reg, -1)
        pred = pred.mean(axis=-1)

        def get_labels(arr):
            index = arr.argmin()
            if arr[index] <= 0.5:
                return names[index]
            return "Cannot Identify"

        labels = [get_labels(p) for p in pred]
        return labels

    def facial_recognition(self, queue, processed_queue):
        try:
            while True:
                frame_list = queue.get()
                if frame_list == "End":
                    break
                # Run face detection and label detected faces
                face_index, face_coords, detected_faces = detect_from_video(frame_list)
                detected_faces = encode_images(detected_faces)
                detected_labels = self.check_registered(detected_faces)

                # Tag labels to individual frames
                face_labels = []
                for indexes in face_index:
                    face_labels.append([detected_labels[index] for index in indexes])
                # print(face_labels)
                # adding borders to frames where face was detected
                frame_list = add_borders(frame_list, face_labels, face_coords, face_index)
                processed_queue.put(frame_list)
                continue

                intruder_count = sum([1 if label == "Cannot Identify" else 0 for label in detected_faces])
                if intruder_count/len(detected_faces) >= 0.2:
                    ## windows_notify("Intruder Detected", "Timestamp?")
                    ## Add "pip install win10toast" in "utils"
                    ## Win10Toast needs to be threaded.
                    ## https://pypi.org/project/win10toast/
                    pass
        except Exception as e:
            print(e)
        finally:
            processed_queue.put("End")

    def start_detecting(self, vid_stream: tuple = None, video_src: str = None):
        """
        Starts detecting motion and recognises a faces of people

        Parameters
        ----------
        vid_stream: A tuple containing cv2.VideoCapture object and a cam_flag
        video_src: Selects the source for cv2.VideoCapture object
        """
        if vid_stream is None:
            vid_stream = (cv2.VideoCapture(0), True) if video_src is None or video_src == "" \
                else (cv2.VideoCapture(video_src), False)
        # TODO remove after saving registered_faces
        if self.register_flag:
            t0 = Thread(target=self.face_register, args = ("Niranjan", record_n_frames(4, vid_stream[0])))
            print("Registering face")
            t0.start()
            t0.join()
            print("Registered new face")
        queue, processed_queue, time_queue, final_queue = Queue(), Queue(), Queue(), Queue()
        stream_fps = int(vid_stream[0].get(cv2.CAP_PROP_FPS))
        t1 = Thread(target=video_stream, args=(queue, vid_stream))
        t2 = Thread(target=motion_detection, args=(queue, processed_queue, time_queue, stream_fps))
        # TODO bug fix fps (may vary from pc)
        t3 = Thread(target=self.facial_recognition, args=(processed_queue, final_queue))
        t4 = Thread(target=save_queue, args=(final_queue, time_queue, stream_fps//3 + 1))

        try:
            t1.start()
            t2.start()
            t3.start()
            t4.start()
            while t2.is_alive():
                pass

        except KeyboardInterrupt:
            print("Stopping the recording")

        finally:
            vid_stream[0].release()
            t1.join()
            t2.join()
            t3.join()
            t4.join()

        # get queue of frames
        # pass queue to detect_faces()
        # get return coordinates list
        # pass co-ordinates list to remove_extra_faces


if __name__ == '__main__':
    HomeSecuritySystem(register=True).start_detecting((cv2.VideoCapture(0), True))

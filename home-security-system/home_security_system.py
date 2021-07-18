import cv2
import numpy as np
from queue import Queue
from threading import Thread
# from win10toast import ToastNotifier

from face_recognition import encode_images, check_similarity
from video_stream import video_stream, save_queue
from motion_detection import motion_detection
from face_detection import detect_from_video

from utils.add_borders import add_borders
from utils.record_n_frames import record_n_frames
from utils.check_param import verify_face_register
from utils.register_face import load_faces, save_faces


class HomeSecuritySystem:
    def __init__(self):
        self.registered_faces = load_faces()

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

        num_reg = len(self.registered_faces)
        names = list(self.registered_faces.keys())
        num_det = len(detected_faces)

        if num_reg == 0:
            return ["Cannot Identify"]*num_det

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
                frame_coords, face_index, detected_faces = detect_from_video(frame_list)
                detected_faces = encode_images(detected_faces)
                detected_labels = self.check_registered(detected_faces)

                # Tag labels to individual frames
                face_labels = []
                for indexes in face_index:
                    face_labels.append([detected_labels[index] for index in indexes])

                # adding borders to frames where face was detected
                frame_list = add_borders(frame_list, frame_coords, face_labels)
                processed_queue.put(frame_list)

                # TODO add notification and timestamp
                '''
                intruder_count = sum([1 if label == "Cannot Identify" else 0 for label in detected_faces])
                intruder_count /= len(detected_faces)
                if intruder_count >= 0.2:
                    toaster = ToastNotifier()
                    import datetime as dt
                    timestamp = dt.datetime.today()
                    toaster.show_toast("Home Security System",
                                       f"Intruder detected at {timestamp:%m/%d, %H:%M:%S}",
                                       threaded=True)
                '''
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

        queue, processed_queue, time_queue, final_queue = Queue(), Queue(), Queue(), Queue()
        stream_fps = int(vid_stream[0].get(cv2.CAP_PROP_FPS))
        t1 = Thread(target=video_stream, args=(queue, vid_stream))
        t2 = Thread(target=motion_detection, args=(queue, processed_queue, time_queue, stream_fps))

        # TODO bug fix fps (may vary from pc)
        t3 = Thread(target=self.facial_recognition, args=(processed_queue, final_queue))
        t4 = Thread(target=save_queue, args=(final_queue, time_queue, stream_fps // 3 + 1))

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


if __name__ == '__main__':
    vid = cv2.VideoCapture("IGNORE/video.mp4")

    my_image = cv2.cvtColor(cv2.resize(cv2.imread("IGNORE/my_image.jpg"), (128, 128)), cv2.COLOR_BGR2RGB)
    my_image = np.array([my_image, my_image, my_image])

    server = HomeSecuritySystem()
    # server.face_register("Nikit", my_image)

    server.start_detecting((vid, True))

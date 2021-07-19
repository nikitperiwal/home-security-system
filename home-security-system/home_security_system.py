import cv2
import numpy as np
from queue import Queue
from threading import Thread

from face_recognition import encode_images, check_similarity
from video_stream import video_stream, save_queue
from motion_detection import motion_detection
from face_detection import detect_from_video

from utils.add_borders import add_borders
from utils.check_param import verify_face_register, verify_hss_args, verify_stream
from utils.register_face import load_faces, save_faces
from utils.notify import create_notification


class HomeSecuritySystem:
    def __init__(self, vid_streams: tuple = None):
        """
        Parameters
        -----------
        vid_streams: a tuple containing VideoCapture objects
        """
        verify_hss_args(vid_streams)
        self.registered_faces = load_faces()
        self.vid_streams = (cv2.VideoCapture(0),) if vid_streams is None else tuple(vid_streams)

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
            return ["Cannot Identify"] * num_det

        labels = []
        for d in detected_faces:
            predictions = []
            for name in names:
                t1 = self.registered_faces[name]
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

    def facial_recognition(self, queue: Queue, processed_queue: Queue):
        """
        Detects faces and passes it for face recognition
        """

        try:
            while True:
                frame_list = queue.get()
                if isinstance(frame_list, str):
                    if frame_list == "End":
                        break
                    elif frame_list == "wait":
                        processed_queue.put("wait")
                        continue

                # Run face detection and label detected faces
                frame_coords, face_index, detected_faces = detect_from_video(frame_list)

                # Runs Face Recognition, if faces are detected.
                if len(detected_faces) > 0:
                    # Getting labels for detected face.
                    detected_faces = encode_images(detected_faces)
                    detected_labels = self.check_registered(detected_faces)

                    # Tag labels to individual frames
                    face_labels = []
                    for indexes in face_index:
                        face_labels.append([detected_labels[index] for index in indexes])

                    # adding borders to frames where face was detected
                    frame_list = add_borders(frame_list, frame_coords, face_labels)

                    intruder_count = sum([1 if "Cannot Identify" in label else 0 for label in face_labels])
                    intruder_count /= len(face_labels)

                    if intruder_count >= 0.4:
                        # TODO Timestamp?
                        import datetime as dt
                        time = dt.datetime.now()
                        message = f"Intruder detected! Please check the video stored at {time: %Y-%m-%d %H:%M:%S}"
                        create_notification("Home Security System", message)

                processed_queue.put(np.array(frame_list))

        except Exception as e:
            print(f"While recognising face exception occurred: \n{e}")
        finally:
            processed_queue.put("End")

    def start_detecting(self, vid_stream: cv2.VideoCapture = None, vid_src: int = None):
        """
        Starts detecting motion and recognises a faces of people

        Parameters
        ----------
        vid_stream: A cv2.VideoCapture object
        vid_src: Selects the source for cv2.VideoCapture object
        """

        verify_stream(vid_stream, vid_src)
        if vid_stream is None:
            vid_stream = cv2.VideoCapture(0) if vid_src is None or vid_src == "" \
                else cv2.VideoCapture(vid_src)

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


if __name__ == '__main__':
    cam_0 = cv2.VideoCapture("IGNORE/video.mp4")
    server = HomeSecuritySystem(vid_streams=(cam_0, ))

    # my_image = cv2.cvtColor(cv2.resize(cv2.imread("IGNORE/my_image.jpg"), (128, 128)), cv2.COLOR_BGR2RGB)
    # my_image = np.array([my_image, my_image, my_image])
    # server.face_register("Nikit", my_image)
    # server.face_register("Niranjan", my_image)

    server.start_camera(0)

import os
import cv2
import numpy as np
from queue import Queue
from utils.check_param import verify_stream_args, verify_save_args


def video_stream(queue: Queue, video_cap: tuple = None):
    """
    Storing the frames from video stream obtained from a cv2.VideoCapture object to a queue

    Parameters
    ----------
    queue: The queue in which the frames are being stored.
    video_cap: A tuple containing cv2.VideoCapture object
    """

    try:
        verify_stream_args(queue, video_cap)
        cap = video_cap
        stream_fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        stop_flag, init_cam = False, True

        # TODO remove after testing
        print('Started video_stream')

        while True:
            for i in range(int(stream_fps)//2):
                ret, frame = cap.read()

                # Return if video ended
                if not ret:
                    stop_flag = True
                    break
                # Executed once when recording is from a camera
                # This is done to prevent a
                elif init_cam:
                    init_cam = False
                    for j in range(10):
                        frame = cap.read()[1]

                frame = cv2.resize(frame, (1280, 720))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            queue.put(np.array(frames))

            frames = []
            if stop_flag:
                break

    except Exception as e:
        print(f"While recording video, exception occurred: \n{e}")

    finally:
        queue.put("End")


def save_queue(processed_queue: Queue, time_queue: Queue,  fps: int = 24):
    """
    Saving a processed queue as a video

    Parameters
    ----------
    processed_queue: The queue from which the frames are taken.
    time_queue: The queue which contains the timestamps when movement was detected
    fps: It's used to vary the frames per second for the saved video
    """

    verify_save_args(processed_queue, time_queue, fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    if not os.path.exists("Detected Videos/"):
        os.makedirs("Detected Videos/")

    try:
        video_path, out = None, None
        while True:
            wait_flag = False
            processed_frames = processed_queue.get()

            if isinstance(processed_frames, str):
                if processed_frames == "End":
                    break
                elif processed_frames == "wait":
                    processed_frames = processed_queue.get()
                    wait_flag = True

            if not wait_flag:
                video_path = "Detected Videos/" + time_queue.get() + ".mp4"
                out = cv2.VideoWriter(video_path, fourcc, fps, (1280, 720))
                # TODO remove after testing
                print("Saving Video")

            for frame in processed_frames:
                out.write(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (1280, 720)))

    except Exception as e:
        print(f"While saving video, exception occurred: \n{e}")

    finally:
        if out is not None:
            out.release()

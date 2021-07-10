import cv2
from queue import Queue
from utils.check_param import verify_stream_args, verify_save_args


def video_stream(queue: Queue, video_cap: tuple = None):
    """
    Storing the frames from video stream obtained from a cv2.VideoCapture object to a queue

    Parameters
    ----------
    queue: The queue in which the frames are being stored.
    video_cap: A tuple containing cv2.VideoCapture object and a cam_flag
    """

    try:
        verify_stream_args(queue, video_cap)
        cap, cam_flag = video_cap
        stream_fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        stop_flag, init_cam = False, True

        while True:
            for i in range(int(stream_fps)*2):
                ret, frame = cap.read()
                # Return if video ended
                if not ret:
                    stop_flag = True
                    break
                # Executed once when recording is form a camer
                elif cam_flag and init_cam:
                    init_cam = False
                    for j in range(10):
                        frame = cap.read()[1]

                frame = cv2.resize(frame, (1280, 720))
                frames.append(frame)
            queue.put(frames)

            frames = []
            if stop_flag:
                break

    except Exception as e:
        print(e)

    finally:
        queue.put("End")


def save_queue(processed_queue: Queue, time_queue: Queue,  fps: int = 30):
    """
    Saving a processed queue as a video

    Parameters
    ----------
    processed_queue: The queue from which the frames are taken.
    time_queue: The queue which contains the timestamps when movement was detected
    fps: It's used to vary the frames per second for the saved video
    """

    try:
        verify_save_args(processed_queue, time_queue, fps)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        while True:
            video_path = "output/" + time_queue.get() + ".mp4"
            out = cv2.VideoWriter(video_path, fourcc, fps, (1280, 720))
            processed_frames = processed_queue.get()
            # TODO remove
            print("Saving Video")
            if processed_frames == "End":
                break
            for i in range(len(processed_frames)):
                out.write(cv2.resize(processed_frames.pop(0), (1280, 720)))

    except Exception as e:
        print(e)

    finally:
        out.release()

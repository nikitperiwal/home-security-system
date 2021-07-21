import cv2
import os
import time
from utils.check_param import verify_motion_args
from utils.remove_file import remove_file

resolution = (1280, 720)


def gaussian_blur(frame):
    """
    Resizes and applies gaussian smoothing on the passed frame and returns the frame
    """

    frame = cv2.resize(frame, (640, 480))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return cv2.GaussianBlur(frame, (21, 21), 0)


def find_contours(init_frame, current_frame, threshold_val, min_contour_area) -> list:
    """
   Finding the difference between 2 images and returning a list of contours detected

   Returns
   -------
   contours: A list of the contours detected in the image
   """

    # Finding the difference between the initial frame and current frame
    frame_delta = cv2.absdiff(init_frame, current_frame)
    thresh = cv2.threshold(frame_delta, threshold_val, 255, cv2.THRESH_BINARY)[1]
    # Dilating the threshold image to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=5)

    # Finding contours in the dilated image
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = contours[-2]

    # removing contours larger than the minimum contour area and returning it
    return [contour for contour in contours if cv2.contourArea(contour) < min_contour_area]


def update_init_frame(init_frame, update_init_thres, counter):
    """
    Checks if the first frame needs to be updated
    """

    if init_frame is None:
        # TODO remove after testing
        print('Started motion_detection')
        return True, counter

    # If the movement lasts for more than threshold, update the initial frame
    if counter['mov'] > update_init_thres:
        counter['mov_total'] += counter['mov']
        counter['mov'] = 0
        return True, counter

    return False, counter


def check_movement(init_frame, frame, counter, movement_flag,
                   threshold_val, min_contour_area) -> (list, bool, list, bool):
    """
    Checks if any movement is detected in the frame with the help of contours

    Parameters
    ----------
    frame: The frame in which we have to detect the movements
    init_frame: The initial frame with which the current frame is compared to
    counter: The counter for the frames
    movement_flag: A boolean indicating whether movement was detected or not
    threshold_val: The thresh value in cv2.threshold.
    min_contour_area: Minimum area of contour.

    Returns
    -------
    counter: Counter for frames
    movement_flag: A boolean indicating whether movement was detected or not
    """

    gray = gaussian_blur(frame)
    contours = find_contours(init_frame, gray, threshold_val, min_contour_area)

    # updating counters
    # If contours exists, movement is detected
    if contours:
        movement_flag = True
        counter['no_mov'] = 0
        counter['mov'] += 1
    # when movement is stopped
    elif movement_flag:
        counter['mov_total'] += counter['mov']
        counter['mov'] = 0
        counter['no_mov'] += 1

    return counter, movement_flag


def motion_detection(vid_stream, vid_index: int, queue, threshold_val: int = 100, min_contour_area: int = 1000,
                     frame_padding: int = -1, update_init_thres: int = 100) -> None:
    """
    Detects motions from the frames in the queue. The frames where motion is detected are added to processed queue

    Parameters
    ----------
    vid_stream: A cv2.VideoCapture object
    vid_index: The index of the camera
    queue: Queue where to put the file names into
    threshold_val: The thresh value in cv2.threshold.
    min_contour_area: Minimum area of contour.
    frame_padding: The number of frame with which the video should be padded.
    update_init_thres: The threshold value before updating the initial frame
    """

    #verify_motion_args(vid_stream, vid_index, queue, threshold_val, min_contour_area, frame_padding, update_init_thres)

    stream_fps = int(vid_stream.get(cv2.CAP_PROP_FPS))
    frame_padding = frame_padding if frame_padding > -1 else stream_fps
    counter = {'mov': 0, 'no_mov': 0, 'mov_total': 0}
    init_frame = None
    mov_flag, first_mov_flag, init_cam_flag = False, True, True
    processed_frames = []
    filepath, out, save_path = None, None, "Motion Videos/" + str(vid_index) + "/"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # TODO index
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    try:
        while True:
            ret, frame = vid_stream.read()

            # Return if video ended
            if not ret:
                break
            # This is executed once to prevent black frames during the initial read
            elif init_cam_flag:
                init_cam_flag = False
                for j in range(10):
                    frame = vid_stream.read()[1]
            frame = cv2.resize(frame, resolution)

            # Updating the initial frame
            init_flag, counter = update_init_frame(init_frame, update_init_thres, counter)
            if init_flag:
                init_frame = gaussian_blur(frame)
                continue

            # Checking for movement
            counter, mov_flag = check_movement(init_frame, frame, counter, mov_flag,
                                               threshold_val, min_contour_area)

            # If movement detected
            if mov_flag:
                # When movement starts
                if first_mov_flag:
                    first_mov_flag = False
                    filepath = save_path + str(time.strftime("%d-%m-%Y-%H-%M-%S", time.localtime())) + ".mp4"
                    # TODO add stream_fps to setting
                    out = cv2.VideoWriter(filepath, fourcc, stream_fps // 3, resolution)
                    print("Saving Video")
                    for i in range(len(processed_frames)):
                        out.write(cv2.resize(processed_frames.pop(0), resolution))
                    processed_frames = []
                # when movement stops
                elif counter['no_mov'] <= frame_padding:
                    out.write(cv2.resize(frame, resolution))

                elif counter['no_mov'] > frame_padding:
                    if counter['mov_total'] < 50:
                        remove_file(filepath)
                    else:
                        queue.put(filepath)
                    mov_flag, init_frame = False, None
                    first_mov_flag = True
                    counter['mov'], counter['no_mov'], counter['mov_total'] = 0, 0, 0
                    # If the video has some unprocessed frames left, run them till all frames are processed
                    print("Waiting for next movement")

            # If no movement is detected
            elif not mov_flag:
                processed_frames.append(frame)
                if len(processed_frames) > frame_padding:
                    processed_frames.pop(0)
            # TODO notify user if movement greater than thres

    except Exception as e:
        print(f"While detecting motion, exception occurred: \n{e}")

    finally:
        if out is not None:
            if counter['mov_total'] > 50:
                queue.put(filepath)
            else:
                remove_file(filepath)
            queue.put("EXIT")
            out.release()

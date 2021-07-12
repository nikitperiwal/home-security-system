import cv2
import time
from queue import Queue
from utils.check_param import verify_motion_args


def gaussian_blur(frame):
    """
    Resizes and applies gaussian smoothing on the passed frame and returns the frame
    """

    frame = cv2.resize(frame, (640, 480))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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


def process_movement(init_frame, frame, processed_queue, time_queue, processed_frames, counter,
                     movement_flag, threshold_val, min_contour_area, frame_padding) -> (list, bool, list):
    """
    Checks if any movement is detected in the frame with the help of contours

    Parameters
    ----------
    frame: The frame in which we have to detect the movements
    init_frame: The initial frame with which the current frame is compared to
    counter: The counter for the frames

    Returns
    -------
    counter: Counter for frames
    movement_flag: A boolean indicating whether movement was detected or not
    processed_frames: A list of processed frames
    """

    gray = gaussian_blur(frame)
    contours = find_contours(init_frame, gray, threshold_val, min_contour_area)

    # If contours exists, movement is detected
    if contours:
        movement_flag = True
        processed_frames.append(frame)
        counter['no_mov'] = 0
        counter['mov'] += 1

    # when movement is stopped
    elif movement_flag:
        # pad frames with 2 frame if save_all_frames is false
        if counter['no_mov'] < frame_padding:
            processed_frames.append(frame)
        counter['mov_total'] += counter['mov']
        counter['mov'] = 0
        counter['no_mov'] += 1
        if counter['no_mov'] >= frame_padding:
            # Save the video and start recording again
            if counter['mov_total'] > 10:
                time_queue.put(str(time.strftime("%d-%m-%Y-%H-%M-%S", time.localtime())))
                processed_queue.put(processed_frames)
            movement_flag, init_frame, processed_frames = False, None, []
            counter['mov'], counter['no_mov'], counter['mov_total'] = 0, 0, 0
            # If the video has some unprocessed frames left, run them till all frames are processed
            print("Waiting for next movement")

    # If no movement is detected
    elif not movement_flag:
        processed_frames.append(frame)
        size = len(processed_frames)
        if size > frame_padding:
            processed_frames.pop(0)

    return counter, movement_flag, processed_frames


def motion_detection(queue: Queue, processed_queue: Queue, time_queue: Queue, stream_fps: int, threshold_val: int = 100,
                     min_contour_area: int = 750, frame_padding: int = -1, update_init_thres: int = 100):
    """
    Detects motions from the frames in the queue. The frames where motion is detected are added to processed queue

    Parameters
    ----------
    queue: Queue containing the list of frames to be processed
    processed_queue: Queue containing the list of frames to be processed
    time_queue: Queue containing the timestamps when motion was detected
    stream_fps:
    threshold_val: The thresh value in cv2.threshold.
    min_contour_area: Minimum area of contour.
    frame_padding: The number of frame with which the video should be padded.
    update_init_thres:
    """

    verify_motion_args(queue, processed_queue, time_queue, threshold_val,
                       min_contour_area, update_init_thres, frame_padding)

    frame_padding = frame_padding if frame_padding > -1 else stream_fps * 2
    counter = {'mov': 0, 'no_mov': 0, 'mov_total': 0}
    init_frame = None
    movement_flag = False
    processed_frames = []

    try:
        while True:
            frames = queue.get()
            if frames == "End":
                break
            # Looping through each frame
            for frame in frames:

                # Updating the initial frame
                init_flag, counter = update_init_frame(init_frame, update_init_thres, counter)
                if init_flag:
                    init_frame = gaussian_blur(frame)
                    continue

                # Checking for movement
                counter, movement_flag, processed_frames = process_movement(init_frame, frame, processed_queue,
                                                                            time_queue, processed_frames, counter,
                                                                            movement_flag, threshold_val,
                                                                            min_contour_area, frame_padding)
                # TODO notify user if movement greater than thres
                if counter['mov_total'] > (2 * stream_fps):
                    pass

    except Exception as e:
        print(e)

    finally:
        if processed_frames:
            time_queue.put(str(time.strftime("%d-%m-%Y-%H-%M-%S", time.localtime())))
            processed_queue.put(processed_frames)
        processed_queue.put("End")

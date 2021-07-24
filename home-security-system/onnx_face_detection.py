import cv2
import time
import numpy as np
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare


def init_model():
    pass


def area_of(left_top, right_bottom):
    """
    Compute the areas of rectangles given two corners.
    Parameters
    -----------
    left_top (N, 2): left top corner.
    right_bottom (N, 2): right bottom corner.

    Returns
    --------
    area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Parameters
    -----------
    boxes0 (N, 4): ground truth boxes.
    boxes1 (N or 1, 4): predicted boxes.
    eps: a small number to avoid 0 as denominator.

    Returns
    --------
    iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Perform hard non-maximum-supression to filter out boxes with iou greater
    than threshold
    Parameters
    -----------
    box_scores (N, 5): boxes in corner-form and probabilities.
    iou_threshold: intersection over union threshold.
    top_k: keep top_k results. If k <= 0, keep all the results.
    candidate_size: only consider the candidates with the highest scores.

    Returns
    --------
    picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    """
    Select boxes that contain human faces

    Parameters
    -----------
    width: original image width
    height: original image height
    confidences (N, 2): confidence array
    boxes (N, 4): boxes array in corner-form
    iou_threshold: intersection over union threshold.
    top_k: keep top_k results. If k <= 0, keep all the results.

    Returns
    --------
    boxes (k, 4): an array of boxes kept
    labels (k): an array of labels for each boxes kept
    probs (k): an array of probabilities for each boxes being in corresponding labels
    """
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
           iou_threshold=iou_threshold,
           top_k=top_k,
           )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


def fix_coords(coords, resolution=(1280, 720), delta=50):
    """ Increases the size of the coordinates """

    x1, y1, w, h = coords
    x1 = max(0, x1 - delta)
    y1 = max(0, y1 - delta)
    x2 = min(resolution[0], x1 + max(128, w + delta))
    y2 = min(resolution[1], y1 + max(128, h + delta))
    return x1, y1, x2, y2


def is_blurred(img, blur_threshold=220):
    """ Returns blur_score if image is blurred, otherwise False """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.convertScaleAbs(cv2.Laplacian(gray, 3))
    score = np.max(score)
    if score < blur_threshold:
        return score
    return False


def extract_face(frame, coords):
    """ Extracts the face from the coords; Returns NONE if face blurred"""

    (x1, y1, x2, y2) = fix_coords(coords)
    face = cv2.resize(frame[y1:y2, x1:x2], (128, 128))
    return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)


def remove_extra_faces(frame_list, coords_list):
    """
    Removes same faces in from sequence of frames and returns face_list, face_index.

    Parameters
    -----------
    frame_list: list containing all the frames
    coords_list: list containing co-ordinates of the faces in the frame

    Returns
    --------
    face_list: list containing the faces in the frame
    face_index: list of face_index that points to faces in face_list
    """

    prev_length = 0
    face_list = []
    face_index = []
    blur_list = []

    for i in range(len(coords_list)):
        if len(coords_list[i]) == 0:
            prev_length = 0
            blur_list = []
            face_index.append([])

        elif len(coords_list[i]) == prev_length:
            # Appends to face_index
            index = range(len(face_list) - len(coords_list[i]), len(face_list))
            index = list(index)
            face_index.append(index)

            # Continues to next frame if no faces blurred.
            if not any(blur_list):
                continue

            # If faces blurred, replace with new faces.
            for j, coords in enumerate(coords_list[i]):
                if blur_list[j]:
                    face = extract_face(frame_list[i], coords)
                    blur_score = is_blurred(face)
                    if blur_score > blur_list[j]:
                        blur_list[j] = blur_score
                        face_list[index[j]] = face

        else:
            # Appends face to face_list and a blur flags to blur list.
            blur_list = []
            for coords in coords_list[i]:
                face = extract_face(frame_list[i], coords)
                blur_list.append(is_blurred(face))
                face_list.append(face)

            # Appends to face_index
            index = range(len(face_list) - len(coords_list[i]), len(face_list))
            index = list(index)
            face_index.append(index)

            prev_length = len(coords_list[i])

    return np.array(face_list), face_index


def detect_from_image(image: np.ndarray):
    h, w, _ = image.shape
    img = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (640, 480))
    img_mean = np.array([127, 127, 127])
    img = (img - img_mean) / 128
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    confidences, boxes = ort_session.run(None, {input_name: img})
    boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)
    return boxes


def detect_from_video(frames):
    coords = []
    for frame in frames:
        coords.append(detect_from_image(frame))

    detected_faces, face_index = remove_extra_faces(frames, coords)
    return coords, face_index, detected_faces

model_path = 'data/models/ultra_light_640.onnx'
model = onnx.load(model_path)
onnx.checker.check_model(model_path)
ort_session = ort.InferenceSession(model_path)
input_name = ort_session.get_inputs()[0].name

if __name__ == '__main__':
    cap = cv2.VideoCapture('IGNORE/output.mp4')
    frame_list = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_list.append(frame)
    detect_from_video(frame_list)

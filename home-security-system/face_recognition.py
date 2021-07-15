import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential


def get_encoder_model():
    """ Returns the encoder after creation and loading weights """

    # Defining the pretrained base model
    pretrained_model = Xception(
        input_shape=(128, 128, 3),
        weights=None,
        include_top=False,
        pooling='avg',
    )

    # Creating the encoder model
    encode_model = Sequential([
        pretrained_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))

    ], name="Encode_Model")

    # Loading the model weights
    encode_model.load_weights("Models/encoder")
    return encode_model


def encode_images(image_list: np.ndarray) -> np.ndarray:
    """
    Returns encodings of a list of images.

    Parameters
    ----------
    image_list: List of RGB images with (?, 128, 128, 3) shape.

    Returns
    -------
    encodings: List of feature vectors
    """

    # Check Statements
    if not isinstance(image_list, np.ndarray):
        raise ValueError("image_list should have d-type: np.ndarray")

    if len(image_list.shape) != 4 or image_list.shape[-3:] != (128, 128, 3):
        raise ValueError("image_list should be of the shape (?, 128, 128, 3)")

    global encoder
    encodings = encoder.predict(image_list)
    return encodings


def check_similarity(tensor_1: np.ndarray, tensor_2: np.ndarray, threshold: float = 1.35) -> np.ndarray:
    """
    Takes two tensors and returns the similarity between them.

    Parameters
    ----------
    tensor_1 : Encodings for Image 1
    tensor_2 : Encodings for Image 2
    threshold: The threshold above which the images are similar

    Returns
    -------
    predictions: Similarity Predictions for the images
    """

    # Check statements
    if not (isinstance(tensor_1, np.ndarray) and isinstance(tensor_1, np.ndarray)):
        raise ValueError("Tensors should have d-type: np.ndarray")
    if not isinstance(threshold, float):
        raise ValueError("threshold should have d-type: float")

    distance = np.sum(np.square(tensor_1 - tensor_2), axis=-1)
    prediction = np.where(distance <= threshold, 0, 1)
    return prediction


# Create the models as global variables
encoder = get_encoder_model()

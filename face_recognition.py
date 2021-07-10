import numpy as np
from tensorflow.keras import backend
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten


def convolution_model():
    """ Defines the Convolution model architecture """

    # Define the tensors for the input image
    image_input = Input((128, 128, 3), name="Input")

    # Defining the pretrained base model
    base_model = Xception(
        input_shape=(128, 128, 3),
        weights=None,
        include_top=False,
        pooling='max',
    )

    model = Sequential([
        base_model,
        Flatten(),
        Dense(2048, activation='sigmoid')
    ], name="Xception")

    # Connect the inputs with the outputs
    return Model(inputs=image_input, outputs=model(image_input), name="Convolution_Model")


def similarity_model():
    """ Defines the Similarity model architecture """

    # Input for the encodings (feature vectors) for the two images.
    encoded_l = Input((2048,), name="Tensor1")
    encoded_r = Input((2048,), name="Tensor2")

    # Add a Subtract layer to compute the absolute difference between the encodings
    l1_layer = Lambda(lambda tensors: backend.abs(tensors[0] - tensors[1]), name='Distance')
    l1_distance = l1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid', name='Prediction')(l1_distance)

    # Connect the inputs with the outputs
    return Model(inputs=[encoded_l, encoded_r], outputs=prediction, name="Similarity_Model")


def get_models():
    """ Returns the models after creation and loading weights """
    backend.clear_session()

    # Creating the model architecture
    model_conv = convolution_model()
    model_similar = similarity_model()

    # Loading the model weights
    model_conv.load_weights("Models/cache_weights")
    model_similar.load_weights("Models/similar_weights")

    return model_conv, model_similar


def convolve_images(image_list: np.ndarray) -> np.ndarray:
    """
    Returns Tensor representation of a list of images.

    Parameters
    ----------
    image_list: List of RGB images with (?, 128, 128, 3) shape.

    Returns
    -------
    image_tensors: List of image tensors
    """

    # Check Statements
    if not isinstance(image_list, np.ndarray):
        raise ValueError("image_list should have d-type: np.ndarray")

    if len(image_list.shape) != 4 or image_list.shape[-3:] != (128, 128, 3):
        raise ValueError("image_list should be of the shape (?, 128, 128, 3)")

    global conv_model
    image_tensors = conv_model.predict(image_list)
    return image_tensors


def check_similarity(tensor_1: np.ndarray, tensor_2: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Takes two tensors and returns the similarity between them.

    Parameters
    ----------
    tensor_1 : Tensor representation for Image 1
    tensor_2 : Tensor representation for Image 2
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

    global similar_model
    predictions = similar_model.predict([tensor_1, tensor_2])
    #predictions = np.where(predictions >= threshold, 1, 0)
    return predictions


# Create the models as global variables
conv_model, similar_model = get_models()

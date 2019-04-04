# State informations to be placed here
import numpy as np
from scipy import misc

actions = []


def preprocess_pong_img(observation: np.ndarray) -> np.ndarray:
    """
    Takes an img observation and does some preprocessing. Crops the image,
    converts it to greyscale, and then scales it down.

    THIS ONLY WORKS FOR PONG.

    :param observation: The image observation, 210 X 160 X 3.
    :return: The greyscale 48 X 48 X 1 image.
    """
    cropped = observation[25:202, :, :]
    gray = np.mean(cropped, axis=2)
    scaled_down = misc.imresize(gray, (48, 48))
    scaled_down = np.reshape(scaled_down, (1, 48, 48, 1))
    return scaled_down


def compress(state, action):
    """
    The model takes both state (usually 2D rectangle) and action (usually a single value) as input, so it is better to arrange them in such a way that
    the combined two becomes a 2D rectangle.
    @params:
        state: The state of the environment
        action: The action to be applied on the given state
    @return:
        c_sa: The 2D rectangle containing information of both state and action
    """
    c_sa = None
    return c_sa

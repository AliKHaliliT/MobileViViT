import tensorflow as tf
from ..activations.mish import Mish
from ..activations.hard_swish import HardSwish


def activation_function(activation: str) -> tf.keras.layers.Layer:

    """

    Method to get the activation function.


    Parameters
    ----------
    activation : str
        Activation function name.
            The options are:
                "relu"
                    Rectified Linear Unit activation function.
                "leaky_relu"
                    Leaky Rectified Linear Unit activation function.
                "hard_swish"
                    Hard Swish activation function.
                "mish"
                    Mish activation function.

    
    Returns
    -------
    activation : tf.keras.layers.Layer
        Activation function.

    """

    if not isinstance(activation, str):
        raise TypeError("activation must be a string")
    if activation not in ["relu", "leaky_relu", "hard_swish", "mish"]:
        raise ValueError("Unknown activation function")
    

    if activation == "relu":
        return tf.keras.layers.ReLU()
    elif activation == "leaky_relu":
        return tf.keras.layers.LeakyReLU()
    elif activation == "hard_swish":
        return HardSwish()
    elif activation == "mish":
        return Mish()
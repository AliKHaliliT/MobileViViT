import tensorflow as tf
from ..layers.sae_3d import SaE3D
from ..layers.ssae_3d import SSaE3D


def squeeze_and_excitation(sae_type: str, reduction_ratio: int = 16, 
                           activation: str = "relu", 
                           omega_0: float = 30.0) -> tf.keras.layers.Layer:

    """

    Method to get the squeeze and excitation layer.

    
    Parameters
    ----------
    sae_type : str
        Squeeze and excitation type.
            The options are:
                "vanilla"
                    Vanilla squeeze and excitation.
                "SSaE"
                    Sine squeeze and excitation.
    
    reduction_ratio : int, optional
        Reduction ratio.

    activation : str, optional
        Activation function.
            The options are:
                "relu"
                    Rectified linear unit.
                "leaky_relu"
                    Leaky rectified linear unit.
                "hard_swish"
                    Hard swish.
                "mish"
                    Mish.

    omega_0 : int or float
        Scaling factor of the Sine activation function. Default value in the original implementation is 30.0.
        See paper Sec. 3.2., final paragraph, and supplement Sec. 1.5. for discussion of omega_0.


    Returns
    -------
    se_layer : tf.keras.layers.Layer
        Squeeze and excitation layer.

    """

    if not isinstance(sae_type, str):
        raise TypeError("se_type must be a string")
    if sae_type not in ["vanilla", "SSE"]:
        raise ValueError("Unknown squeeze and excitation type")
    if not isinstance(reduction_ratio, int):
        raise TypeError("reduction_ratio must be an integer")
    if reduction_ratio < 0:
        raise ValueError("reduction_ratio must be positive")
    if not isinstance(activation, str):
        raise TypeError("activation must be a string")
    if activation not in ["relu", "leaky_relu", "hard_swish", "mish"]:
        raise ValueError("Unknown activation function")
    if not isinstance(omega_0, int) and not isinstance(omega_0, float):
        raise TypeError("omega_0 must be an integer or a float")


    if sae_type == "vanilla":
        return SaE3D(reduction_ratio=reduction_ratio, activation=activation)
    elif sae_type == "SSaE":
        return SSaE3D(reduction_ratio=reduction_ratio, omega_0=omega_0)
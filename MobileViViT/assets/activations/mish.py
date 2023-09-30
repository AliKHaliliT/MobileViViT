import tensorflow as tf
from typing import Any


class Mish(tf.keras.layers.Layer):

    """

    Mish activation function from the paper "Mish: A Self Regularized Non-Monotonic Activation Function"
    Link: https://arxiv.org/abs/1908.08681

    """

    def __init__(self, **kwargs) -> None:

        """

        Constructor of the Mish activation function.
        
        
        Parameters
        ----------
        None.

        
        Returns
        -------
        None.
        
        """
        
        super().__init__(**kwargs)


    def build(self, input_shape: tf.TensorShape) -> None:

        """

        Build method of the Mish activation function.


        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of the input tensor.


        Returns
        -------
        None.
        
        """

        if not isinstance(input_shape, tf.TensorShape):
            raise TypeError("Input shape must be a TensorShape object")
        

        super().build(input_shape)
    

    def call(self, X: tf.Tensor) -> tf.Tensor:

        """

        Call method of the Mish activation function.


        Parameters
        ----------
        X : tf.Tensor
            Input tensor.

        
        Returns
        -------
        X_transformed : tf.Tensor
            Output tensor.
        
        """


        return tf.multiply(X, tf.tanh(tf.keras.activations.softplus(X)))
    

    def get_config(self) -> dict[str, Any]:

        """

        Method to get the configuration of the Mish activation function.


        Parameters
        ----------
        None.


        Returns
        -------
        config : dict
            Configuration of the Mish activation function.

        """
            
        config = super().get_config()


        return config
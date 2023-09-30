# Importing the libraries
import tensorflow as tf
from .fc_layer import FCLayer
from math import ceil
from typing import Any


class SaE3D(tf.keras.layers.Layer):

    """

    Squeeze and Excitation layer adapted for higher dimensional tasks from the paper "Squeeze-and-Excitation Networks"
    Link: https://arxiv.org/abs/1709.01507

    """
    
    def __init__(self, reduction_ratio: int = 16, activation: str = "relu", **kwargs) -> None:

        """

        Constructor of the SaE3D layer.


        Parameters
        ----------
        reduction_ratio : int, optional
            Reduction ratio of the squeeze operation. Default value in the original implementation is 16.

        activation : str, optional
            Activation function of the FC layers. Default value is "relu".
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
        None.

        """

        if not isinstance(reduction_ratio, int):
            raise TypeError("reduction_ratio must be an integer")
        if reduction_ratio < 0:
            raise ValueError("reduction_ratio must be positive")
        if not isinstance(activation, str):
            raise TypeError("activation must be a string")
        if activation not in ["relu", "leaky_relu", "hard_swish", "mish"]:
            raise ValueError("Unknown activation function")


        super().__init__(**kwargs)

        self.reduction_ratio = reduction_ratio
        self.activation = activation


    def build(self, input_shape: tf.TensorShape) -> None:

        """

        Build method of the SaE3D layer.


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

        self.channels = input_shape[-1]
        self.reduce = tf.keras.layers.GlobalAveragePooling3D()
        self.dense = FCLayer(units=ceil(self.channels // self.reduction_ratio), activation=self.activation, dropout=0.0)
        self.dense_1 = FCLayer(units=self.channels, activation=self.activation, dropout=0.0)
        self.reshape = tf.keras.layers.Reshape((1, 1, 1, -1))
        self.multiply = tf.keras.layers.Multiply()


    def call(self, X: tf.Tensor) -> tf.Tensor:

        """

        Call method of the SaE3D layer.


        Parameters
        ----------
        X : tf.Tensor
            Input tensor.
        
            
        Returns
        -------
        X_transformed : tf.Tensor
            Output tensor.

        """


        return self.multiply([X, self.reshape(self.dense_1(self.dense(self.reduce(X))))])
    

    def get_config(self) -> dict[str, Any]:

        """

        Method to get the configuration of the SaE3D layer.


        Parameters
        ----------
        None.


        Returns
        -------
        config : dict
            Configuration of the SaE3D layer.

        """
            
        config = super().get_config()

        config.update({
            "reduction_ratio": self.reduction_ratio,
            "activation": self.activation
        })


        return config
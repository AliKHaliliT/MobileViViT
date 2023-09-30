# Importing the libraries
import tensorflow as tf
from typing import Union, Any
from .sine_layer import SineLayer
from math import ceil


class SSaE3D(tf.keras.layers.Layer):

    """

    Modified Squeeze and Excitation layer using a Sine layer adapted for higher dimensional tasks from 
    the papers "Squeeze-and-Excitation Networks" and "Implicit Neural Representations with Periodic Activation Functions"
    Links: https://arxiv.org/abs/1709.01507 and https://arxiv.org/abs/2006.09661


    """
    
    def __init__(self, reduction_ratio: int = 16, omega_0: Union[int, float] = 30.0, **kwargs) -> None:

        """

        Constructor of the SSaE3D layer.


        Parameters
        ----------
        reduction_ratio : int, optional
            Reduction ratio of the squeeze operation. Default value in the original implementation is 16.

        omega_0 : int or float, optional
            Scaling factor of the Sine activation function. Default value in the original implementation is 30.0.
            See paper Sec. 3.2., final paragraph, and supplement Sec. 1.5. for discussion of omega_0.

        
        Returns
        -------
        None.

        """

        if not isinstance(reduction_ratio, int):
            raise TypeError("reduction_ratio must be an integer")
        if reduction_ratio < 0:
            raise ValueError("reduction_ratio must be positive")
        if not isinstance(omega_0, int) and not isinstance(omega_0, float):
            raise TypeError("omega_0 must be an integer or a float")


        super().__init__(**kwargs)

        self.reduction_ratio = reduction_ratio
        self.omega_0 = omega_0


    def build(self, input_shape: tf.TensorShape) -> None:

        """

        Build method of the SSaE3D layer.


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
        self.dense = SineLayer(units=ceil(self.channels // self.reduction_ratio), is_first=True, omega_0=self.omega_0)
        self.dense_1 = SineLayer(units=self.channels, is_first=False, omega_0=self.omega_0)
        self.reshape = tf.keras.layers.Reshape((1, 1, 1, -1))
        self.multiply = tf.keras.layers.Multiply()


    def call(self, X: tf.Tensor) -> tf.Tensor:

        """

        Call method of the SSaE3D layer.


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

        Method to get the configuration of the SSaE3D layer.


        Parameters
        ----------
        None.


        Returns
        -------
        config : dict
            Configuration of the SSaE3D layer.

        """
            
        config = super().get_config()

        config.update({
            "reduction_ratio": self.reduction_ratio,
            "omega_0": self.omega_0
        })


        return config
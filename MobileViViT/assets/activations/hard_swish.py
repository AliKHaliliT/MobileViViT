import tensorflow as tf
from typing import Any


class HardSwish(tf.keras.layers.Layer):

    """

    HardSwish activation function from the paper "Searching for MobileNetV3"
    Link: https://arxiv.org/abs/1905.02244

    """

    def __init__(self, **kwargs) -> None:

        """

        Constructor of the HardSwish activation function.
        
        
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

        Build method of the HardSwish activation function.


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

        Call method of the HardSwish activation function.


        Parameters
        ----------
        X : tf.Tensor
            Input tensor.

        
        Returns
        -------
        X_transformed : tf.Tensor
            Output tensor.
        
        """


        return tf.multiply(X, tf.divide(tf.nn.relu6(tf.add(X, 3.0)), 6.0))
    

    def get_config(self) -> dict[str, Any]:

        """

        Method to get the configuration of the HardSwish activation function.


        Parameters
        ----------
        None.


        Returns
        -------
        config : dict
            Configuration of the HardSwish activation function.

        """
            
        config = super().get_config()


        return config
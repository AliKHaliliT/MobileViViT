import tensorflow as tf
from typing import Union, Any


class Sine(tf.keras.layers.Layer):

    """

    Sine activation function from the paper "Implicit Neural Representations with Periodic Activation Functions"
    Link: https://arxiv.org/abs/2006.09661

    """

    def __init__(self, omega_0: Union[int, float], **kwargs) -> None:

        """

        Constructor of the Sine activation function.
        
        
        Parameters
        ----------
        omega_0 : int or float
            Scaling factor of the Sine activation function. Default value in the original implementation is 30.0. 
            See paper Sec. 3.2., final paragraph, and supplement Sec. 1.5. for discussion of omega_0.

        
        Returns
        -------
        None.
        
        """

        if not isinstance(omega_0, int) and not isinstance(omega_0, float):
            raise TypeError("omega_0 must be an integer or a float")
        
        
        super().__init__(**kwargs)

        self.omega_0 = omega_0


    def build(self, input_shape: tf.TensorShape) -> None:

        """

        Build method of the Sine activation function.


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

        Call method of the Sine activation function.


        Parameters
        ----------
        X : tf.Tensor
            Input tensor.

        
        Returns
        -------
        X_transformed : tf.Tensor
            Output tensor.
        
        """


        return tf.math.sin(tf.multiply(self.omega_0, X))
    

    def get_config(self) -> dict[str, Any]:

        """

        Method to get the configuration of the Sine activation function.


        Parameters
        ----------
        None.


        Returns
        -------
        config : dict
            Configuration of the Sine activation function.

        """
            
        config = super().get_config()

        config.update({
            "omega_0": self.omega_0
        })


        return config
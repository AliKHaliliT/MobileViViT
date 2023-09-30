import tensorflow as tf
from typing import Union, Optional, Any
from ..utils.sine_layer_initializer import SineLayerInitializer
from ..activations.sine import Sine


class SineLayer(tf.keras.layers.Layer):

    """

    Slightly modified implementation of the Sine layer from 
    the paper "Implicit Neural Representations with Periodic Activation Functions"
    Link: https://arxiv.org/abs/2006.09661

    """

    def __init__(self, units: int, is_first: bool, omega_0: Union[int, float], 
                 dropout: Optional[float] = None, **kwargs) -> None:

        """

        Constructor of the Sine layer.


        Parameters
        ----------
        units : int
            Number of units in the Sine layer.

        is_first : bool
            Whether the layer is the first layer of the network.
        
        omega_0 : int or float
            Scaling factor of the Sine activation function. Default value in the original implementation is 30.0.
            See paper Sec. 3.2., final paragraph, and supplement Sec. 1.5. for discussion of omega_0.
        
        dropout : float, optional
            Fraction of the input units to drop. It must be between 0 and 1. 
            Default value is None as in the original implementation.

            
        Returns
        -------
        None.
        
        """

        if not isinstance(units, int):
            raise TypeError("units must be an integer")
        if units < 0:
            raise ValueError("units must be positive")
        if not isinstance(is_first, bool):
            raise TypeError("is_first must be a boolean")
        if not isinstance(omega_0, int) and not isinstance(omega_0, float):
            raise TypeError("omega_0 must be an integer or a float")
        if dropout is not None:
            if not isinstance(dropout, float):
                raise TypeError("dropout rate must be a float")
            if dropout < 0 or dropout > 1:
                raise ValueError("dropout rate must be between 0 and 1")
        

        super().__init__(**kwargs)

        self.units = units
        self.is_first = is_first
        self.omega_0 = omega_0
        self.dropout = dropout
        

    def build(self, input_shape: tf.TensorShape) -> None:

        """

        Build method of the Sine layer.


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

        self.dense = tf.keras.layers.Dense(self.units, kernel_initializer=SineLayerInitializer(self.units, self.is_first, self.omega_0))
        self.activate = Sine(self.omega_0)
        if self.dropout is not None:
            self.drop = tf.keras.layers.Dropout(self.dropout)


    def call(self, X: tf.Tensor) -> tf.Tensor:

        """
        
        Call method of the Sine layer.


        Parameters
        ----------
        X : tf.Tensor
            Input tensor.

    
        Returns
        -------
        X_transformed : tf.Tensor
            Output tensor.

        """


        return self.drop(self.activate(self.dense(X))) if self.dropout is not None else self.activate(self.dense(X))
    

    def get_config(self) -> dict[str, Any]:

        """

        Method to get the configuration of the Sine layer.


        Parameters
        ----------
        None.


        Returns
        -------
        config : dict
            Configuration of the Sine layer.

        """
            
        config = super().get_config()

        config.update({
            "units": self.units,
            "is_first": self.is_first,
            "omega_0": self.omega_0,
            "dropout": self.dropout
        })


        return config
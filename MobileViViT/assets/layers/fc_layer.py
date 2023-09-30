import tensorflow as tf
from ..utils.activation_function import activation_function
from typing import Any


class FCLayer(tf.keras.layers.Layer):

    """

    A simple custom Fully Connected layer.

    """

    def __init__(self, units: int, activation: str, dropout: float, **kwargs) -> None:

        """

        Constructor of the FC layer.
        
        
        Parameters
        ----------
        units : int
            Number of units in the FC layer.

        activation : str
            Activation function of the FC layer.
                The options are:
                    "relu"
                        Rectified Linear Unit activation function.
                    "leaky_relu"
                        Leaky Rectified Linear Unit activation function.
                    "hard_swish"
                        Hard Swish activation function.
                    "mish"
                        Mish activation function.

        dropout : float
            Fraction of the input units to drop in the FC layer. It must be between 0 and 1.

        
        Returns
        -------
        None.
        
        """

        if not isinstance(units, int):
            raise TypeError("units must be an integer")
        if units < 0:
            print(units)
            raise ValueError("units must be positive")
        if not isinstance(activation, str):
            raise TypeError("activation must be a string")
        if activation not in ["relu", "leaky_relu", "hard_swish", "mish"]:
            raise ValueError("Unknown activation function")
        if not isinstance(dropout, float):
            raise TypeError("dropout rate must be a float")
        if dropout < 0 or dropout > 1:
            raise ValueError("dropout rate must be between 0 and 1")
        

        super().__init__(**kwargs)

        self.units = units
        self.activation = activation
        self.dropout = dropout


    def build(self, input_shape: tf.TensorShape) -> None:

        """

        Build method of the FC layer.


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

        self.dense = tf.keras.layers.Dense(self.units)
        self.activate = activation_function(self.activation)
        self.drop = tf.keras.layers.Dropout(self.dropout)
        

    def call(self, X: tf.Tensor) -> tf.Tensor:

        """

        Call method of the FC layer.


        Parameters
        ----------
        X : tf.Tensor
            Input tensor.
        
            
        Returns
        -------
        X_transformed : tf.Tensor
            Output tensor.

        """
        

        return self.drop(self.activate(self.dense(X)))
    

    def get_config(self) -> dict[str, Any]:

        """

        Method to get the configuration of the FC layer.


        Parameters
        ----------
        None.


        Returns
        -------
        config : dict
            Configuration of the FC layer.

        """
            
        config = super().get_config()

        config.update({
            "units": self.units,
            "activation": self.activation,
            "dropout": self.dropout
        })


        return config
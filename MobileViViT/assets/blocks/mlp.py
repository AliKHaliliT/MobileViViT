import tensorflow as tf
from ..layers.fc_layer import FCLayer
from typing import Any


class MLP(tf.keras.layers.Layer):

    """

    A Simple MLP block.

    """

    def __init__(self, units_list: list[int], activation: str = "hard_swish", 
                 dropout: float = 0.1, **kwargs) -> None:

        """

        Constructor of the MLP block.
        
        
        Parameters
        ----------
        units_list : list
            List of number of units in each FC layer.

        activation : str, optional
            Activation function of the FC layers. The default is "hard_swish".
                The options are:
                    "relu"
                        Rectified Linear Unit activation function.
                    "leaky_relu"
                        Leaky Rectified Linear Unit activation function.
                    "hard_swish"
                        Hard Swish activation function.
                    "mish"
                        Mish activation function.

        dropout : float, optional
            Fraction of the input units to drop in the FC layers. It must be between 0 and 1. The default is 0.1.

        
        Returns
        -------
        None.
        
        """

        if not isinstance(units_list, list):
            raise TypeError("units_list must be a list")
        if len(units_list) < 1:
            raise ValueError("units_list must contain at least one element")
        if not all(isinstance(units, int) for units in units_list):
            raise TypeError("units_list must contain only integers")
        if not all(units > 0 for units in units_list):
            raise ValueError("units_list must contain only positive integers")
        if not isinstance(activation, str):
            raise TypeError("activation must be a string")
        if activation not in ["relu", "leaky_relu", "hard_swish", "mish"]:
            raise ValueError("Unknown activation function")
        if not isinstance(dropout, float):
            raise TypeError("dropout rate must be a float")
        if dropout < 0 or dropout > 1:
            raise ValueError("dropout rate must be between 0 and 1")
        

        super().__init__(**kwargs)

        self.units_list = units_list
        self.activation = activation
        self.dropout = dropout


    def build(self, input_shape: tf.TensorShape) -> None:

        """

        Build method of the MLP block.


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

        self.mlp = [FCLayer(units, self.activation, self.dropout) for units in self.units_list]
        

    def call(self, X: tf.Tensor) -> tf.Tensor:

        """

        Call method of the MLP block.


        Parameters
        ----------
        X : tf.Tensor
            Input tensor.
        
            
        Returns
        -------
        X_transformed : tf.Tensor
            Output tensor.

        """

        for layer in self.mlp:
            X = layer(X)
        

        return X
    

    def get_config(self) -> dict[str, Any]:

        """

        Method to get the configuration of the MLP block.


        Parameters
        ----------
        None.


        Returns
        -------
        config : dict
            Configuration of the MLP block.

        """
            
        config = super().get_config()

        config.update({
            "units_list": self.units_list,
            "activation": self.activation,
            "dropout": self.dropout
        })


        return config
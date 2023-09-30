import tensorflow as tf
from typing import Union, Optional, Any
from ..layers.sine_layer import SineLayer


class SIREN(tf.keras.layers.Layer):

    """

    Slightly modified implementation of the SIREN from 
    the paper "Implicit Neural Representations with Periodic Activation Functions"
    Link: https://arxiv.org/abs/2006.09661

    """

    def __init__(self, units_list: list[int], omega_0: Union[int, float] = 30.0, 
                 dropout: Optional[float] = None, **kwargs) -> None:

        """

        Constructor of the SIREN block.
        
        
        Parameters
        ----------
        units_list : list
            List of number of units in each Sine layer.

        omega_0 : int or float, optional
            Scaling factor of the Sine activation function. Default value in the original implementation is 30.0. 
            See paper Sec. 3.2., final paragraph, and supplement Sec. 1.5. for discussion of omega_0.

        dropout : float, optional
            Fraction of the input units to drop. It must be between 0 and 1. 
            Default value is None as in the original implementation.

        
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
        if not isinstance(omega_0, int) and not isinstance(omega_0, float):
            raise TypeError("omega_0 must be an integer or a float")
        if dropout is not None:
            if not isinstance(dropout, float):
                raise TypeError("dropout rate must be a float")
            if dropout < 0 or dropout > 1:
                raise ValueError("dropout rate must be between 0 and 1")
        

        super().__init__(**kwargs)

        self.units_list = units_list
        self.omega_0 = omega_0
        self.dropout = dropout


    def build(self, input_shape: tf.TensorShape) -> None:

        """

        Build method of the SIREN block.


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

        self.siren = [SineLayer(units=units, is_first=(index == 0), omega_0=self.omega_0, 
                                dropout=self.dropout if self.dropout else 0.0) 
                                for index, units in enumerate(self.units_list)]

        
    def call(self, X: tf.Tensor) -> tf.Tensor:

        """

        Call method of the SIREN block.


        Parameters
        ----------
        X : tf.Tensor
            Input tensor.
        
            
        Returns
        -------
        X_transformed : tf.Tensor
            Output tensor.

        """

        for layer in self.siren:
            X = layer(X)
        

        return X
    

    def get_config(self) -> dict[str, Any]:

        """

        Method to get the configuration of the SIREN block.


        Parameters
        ----------
        None.


        Returns
        -------
        config : dict
            Configuration of the SIREN block.

        """
            
        config = super().get_config()

        config.update({
            "units_list": self.units_list,
            "omega_0": self.omega_0,
            "dropout": self.dropout
        })


        return config
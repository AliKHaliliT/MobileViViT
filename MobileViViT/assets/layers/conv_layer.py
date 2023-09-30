from typing import Optional, Any
import tensorflow as tf
from .conv2plus1d import Conv2Plus1D
from ..utils.activation_function import activation_function


class ConvLayer(tf.keras.layers.Layer):

    """
    
    A simple custom Conv layer containing Conv2Plus1D.
    
    """

    def __init__(self, filters: int, kernel_size: tuple[int, int, int], 
                 strides: tuple[int, int, int], padding: str, batch_norm: bool, 
                 activation: Optional[str] = None, **kwargs) -> None:

        """

        Constructor of the Conv layer.
        
        
        Parameters
        ----------
        filters : int
            Number of filters in the convolutional layer.

        kernel_size : tuple
            Kernel size of the convolutional layer.

        strides : tuple
            Strides of the convolutional layer.

        padding : str
            Padding of the convolutional layer.
                The options are:
                    "valid"
                        No padding.
                    "same"
                        Padding with zeros.

        batch_norm : bool
            Whether to use batch normalization.

        activation : str, optional
            Activation function of the layer. The default value is None.
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

        if not isinstance(filters, int):
            raise TypeError("filters must be an integer")
        if filters < 0:
            raise ValueError("filters must be positive")
        if not isinstance(kernel_size, tuple):
            raise TypeError("kernel_size must be a tuple")
        if len(kernel_size) != 3:
            raise ValueError("kernel_size must be a tuple of length 3")
        if not all(isinstance(k, int) for k in kernel_size):
            raise TypeError("kernel_size must contain only integers")
        if not all([k > 0 for k in kernel_size]):
            raise ValueError("kernel_size must contain only positive integers")
        if not isinstance(strides, tuple):
            raise TypeError("strides must be a tuple")
        if len(strides) != 3:
            raise ValueError("strides must be a tuple of length 3")
        if not all(isinstance(s, int) for s in strides):
            raise TypeError("strides must contain only integers")
        if not all([s > 0 for s in strides]):
            raise ValueError("strides must contain only positive integers")
        if not isinstance(padding, str):
            raise TypeError("padding must be a string")
        if padding not in ["valid", "same"]:
            raise ValueError("Unknown padding type")
        if not isinstance(batch_norm, bool):
            raise TypeError("batch_norm must be a boolean")
        if activation is not None:
            if not isinstance(activation, str):
                raise TypeError("activation must be a string")
            if activation not in ["relu", "leaky_relu", "hard_swish", "mish"]:
                raise ValueError("Unknown activation function")


        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.batch_norm = batch_norm
        self.activation = activation


    def build(self, input_shape: tf.TensorShape) -> None:

        """

        Build method of the Conv layer.


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

        self.convolution = Conv2Plus1D(filters=self.filters, kernel_size=self.kernel_size, 
                                       strides=self.strides, padding=self.padding)
        if self.batch_norm:
            self.batch_normalize = tf.keras.layers.BatchNormalization()
        if self.activation is not None:
            self.activate = activation_function(self.activation)
                                

    def call(self, X: tf.Tensor) -> tf.Tensor:

        """

        Call method of the Conv layer.


        Parameters
        ----------
        X : tf.Tensor
            Input tensor.
        
            
        Returns
        -------
        X_transformed : tf.Tensor
            Output tensor.

        """

        X_transformed = self.convolution(X)
        if self.batch_norm:
            X_transformed = self.batch_normalize(X_transformed)


        return self.activate(X_transformed) if self.activation is not None else X_transformed


    def get_config(self) -> dict[str, Any]:

        """

        Method to get the configuration of the Conv layer.


        Parameters
        ----------
        None.


        Returns
        -------
        config : dict
            Configuration of the Conv layer.

        """
            
        config = super().get_config()

        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "batch_norm": self.batch_norm,
            "activation": self.activation,
        })


        return config
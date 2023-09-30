import tensorflow as tf
from math import ceil
from typing import Any


class Conv2Plus1D(tf.keras.layers.Layer):

    """
    
    Conv2Plus1D layer from the paper "A Closer Look at Spatiotemporal Convolutions for Action Recognition"
    Link: https://arxiv.org/abs/1711.11248v3
    
    """

    def __init__(self, filters: int, kernel_size: tuple[int, int, int], 
                 strides: tuple[int, int, int], padding: str, **kwargs) -> None:

        """

        Constructor of the Conv2Plus1D layer.
        
        
        Parameters
        ----------
        filters : int
            Number of filters in the temporal convolutional layer.
            The number of filters in the spatial decomposition is calculated based this value.
            See paper Sec. 3.5. for more details.

        kernel_size : tuple
            Kernel size of the convolutional layers.

        strides : tuple
            Strides of the convolutional layers.

        padding : str
            Padding of the convolutional layers.
                The options are:
                    "valid"
                        No padding.
                    "same"
                        Padding with zeros.

        
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


        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding


    def build(self, input_shape: tf.TensorShape) -> None:

        """

        Build method of the Conv2Plus1D layer.


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

        # Calcualting the number of filters required for the spatial decomposition
        spatial_filters = ceil((self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * input_shape[-1] * self.filters) /
                               ((self.kernel_size[1] * self.kernel_size[2] * input_shape[-1]) + (self.kernel_size[0] * self.filters)))
        # Spatial decomposition
        self.spatial_decompose = tf.keras.layers.Conv3D(filters=spatial_filters, 
                                                        kernel_size=(1, self.kernel_size[1], self.kernel_size[2]),
                                                        strides=(1, self.strides[1], self.strides[2]), 
                                                        padding=self.padding)
        
        # Temporal decomposition
        self.temporal_decompose = tf.keras.layers.Conv3D(filters=self.filters, 
                                                         kernel_size=(self.kernel_size[0], 1, 1), 
                                                         strides=(self.strides[0], 1, 1), 
                                                         padding=self.padding)
                                

    def call(self, X: tf.Tensor) -> tf.Tensor:

        """

        Call method of the Conv2Plus1D layer.


        Parameters
        ----------
        X : tf.Tensor
            Input tensor.
        
            
        Returns
        -------
        X_transformed : tf.Tensor
            Output tensor.

        """


        return self.temporal_decompose(self.spatial_decompose(X))


    def get_config(self) -> dict[str, Any]:

        """

        Method to get the configuration of the Conv2Plus1D layer.


        Parameters
        ----------
        None.


        Returns
        -------
        config : dict
            Configuration of the Conv2Plus1D layer.

        """
            
        config = super().get_config()

        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding
        })


        return config
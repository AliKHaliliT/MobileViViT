
# Importing the libraries
import tensorflow as tf
from typing import Optional
from ..layers.conv_layer import ConvLayer
from ..utils.squeeze_and_excitation import squeeze_and_excitation
from typing import Any


class MVNBlock(tf.keras.layers.Layer):

    """

    Slighlty modified implementation of the MobileNetV2 block adapted for higher dimensional tasks from 
    the paper "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
    Link: https://arxiv.org/abs/1801.04381

    Some of the modifications are:

        1.  A Conv2Plus1D layer from the paper "A Closer Look at Spatiotemporal Convolutions for Action Recognition"
            Link: https://arxiv.org/abs/1711.11248v3

        2.  A Squeeze and Excitation layer from the paper "Squeeze-and-Excitation Networks"
            Link: https://arxiv.org/abs/1709.01507

        3.  A slightly modified implementation of the vanilla Squeeze and Excitation layer using the Sine layer from 
            the paper "Implicit Neural Representations with Periodic Activation Functions"
            Link: https://arxiv.org/abs/2006.09661

        4.  A slightly modified implementation of the residual connection of the ResNet-D block from
            the paper "Bag of Tricks for Image Classification with Convolutional Neural Networks"
            Link: https://arxiv.org/abs/1812.01187

    """

    def __init__(self, expansion_filters: int, filters: int, 
                 kernel_size: tuple[int, int, int], strides: tuple[int, int, int], 
                 padding: str, activation: str, SaE: Optional[str] = None, **kwargs) -> None:
        
        """

        Constructor of the MVNBlock.


        Parameters
        ----------
        expansion_filters : int
            Number of filters in the expansion convolutional layers.

        filters : int
            Number of filters in the convolutional layers.

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

        activation : str
            Activation function of the convolutional layers. The default is "hard_swish".
                The options are:
                    "relu"
                        Rectified Linear Unit activation function.
                    "leaky_relu"
                        Leaky Rectified Linear Unit activation function.
                    "hard_swish"
                        Hard Swish activation function.
                    "mish"
                        Mish activation function.

        SaE : str
            Type of Squeeze and Excitation layer to use. The default is None.
                The options are:
                    "vanilla"
                        Vanilla squeeze and excitation.
                    "SSaE"
                        Sine squeeze and excitation.

        
        Returns
        -------
        None.
        
        """

        if not isinstance(expansion_filters, int):
            raise TypeError("expansion_filters must be an integer")
        if expansion_filters < 0:
            raise ValueError("expansion_filters must be positive")
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
        if not isinstance(activation, str):
            raise TypeError("activation must be a string")
        if activation not in ["relu", "leaky_relu", "hard_swish", "mish"]:
            raise ValueError("Unknown activation function")
        if SaE is not None:
            if not isinstance(SaE, str):
                raise TypeError("SaE must be a string")
            if SaE not in ["vanilla", "SSaE"]:
                raise ValueError("Unknown SaE type")


        super().__init__(**kwargs)

        self.expansion_filters = expansion_filters
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.SaE = SaE


    def build(self, input_shape: tf.TensorShape) -> None:

        """

        Build method of the MVNBlock.


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

        # Main Path
        self.expand = ConvLayer(filters=self.expansion_filters, kernel_size=(1, 1, 1),
                                strides=(1, 1, 1), padding="valid", batch_norm=True, 
                                activation=self.activation)

        self.feature_extract = ConvLayer(filters=self.expansion_filters, kernel_size=self.kernel_size,
                                         strides=self.strides, padding=self.padding, batch_norm=True,
                                         activation=self.activation)

        if self.SaE is not None:
            self.squeeze_and_excite = squeeze_and_excitation(self.SaE)

        self.finalize = ConvLayer(filters=self.filters, kernel_size=(1, 1, 1),
                                  strides=(1, 1, 1), padding="valid", batch_norm=True, 
                                  activation=None)

        # Residual Path
        if input_shape[-1] == self.filters: 

            self.reduce = tf.keras.layers.AveragePooling3D(pool_size=(1, 2, 2), strides=self.strides, padding=self.padding)
            self.skip_feature_extract = ConvLayer(filters=self.filters, kernel_size=(1, 1, 1),
                                                  strides=(1, 1, 1), padding="valid", batch_norm=False, 
                                                  activation=None)
            self.add = tf.keras.layers.Add()


    def call(self, X: tf.Tensor) -> tf.Tensor:

        """

        Call method of the MVNBlock.


        Parameters
        ----------
        X : tf.Tensor
            Input tensor.
        
            
        Returns
        -------
        X_transformed : tf.Tensor
            Output tensor.

        """

        X_transformed = self.expand(X)

        X_transformed = self.feature_extract(X_transformed)

        if self.SaE is not None:
            X_transformed = self.squeeze_and_excite(X_transformed)

        X_transformed = self.finalize(X_transformed)

        if X.shape[-1] == X_transformed.shape[-1]:
            X_transformed = self.add([self.skip_feature_extract(self.reduce(X)), X_transformed])


        return X_transformed
    

    def get_config(self) -> dict[str, Any]:

        """

        Method to get the configuration of the MVNBlock.


        Parameters
        ----------
        None.


        Returns
        -------
        config : dict
            Configuration of the MVNBlock.

        """
            
        config = super().get_config()

        config.update({
            "expansion_filters": self.expansion_filters,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "activation": self.activation,
            "SaE": self.SaE
        })


        return config
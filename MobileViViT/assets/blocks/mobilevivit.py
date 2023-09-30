
import tensorflow as tf
from ..layers.conv_layer import ConvLayer
from ..layers.unfold import Unfold
from .transformer import Transformer
from ..layers.fold import Fold
from typing import Any


class MobileViViT(tf.keras.layers.Layer):

    """
    
    Slighlty modified implementation of the MobileViT block adapted for higher dimensional tasks from 
    the paper "MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer"
    Link: https://arxiv.org/abs/2110.02178
    
    """

    def __init__(self, projection_dim: int, kernel_size: list[tuple[int, int, int]], 
                 strides: tuple[int, int, int], padding: str, patch_size: tuple[int, int, int], 
                 num_transformer_layers: int, activation: str = "hard_swish", 
                 num_transformer_heads: int = 2, dropout_mha: float = 0.1, 
                 aproximator_type: str = "vanilla", dropout_mlp: float = 0.1, **kwargs) -> None:
        
        """
        
        Constructor of the MobileViViT block.


        Parameters
        ----------
        projection_dim : int
            Dimension of the projection sapce.

        kernel_size : tuple
            Kernel size of the convolutional layers.
            The first element is the kernel size of the local feature extractor layers.
            The second element is the kernel size of the fuser layer.

        strides : tuple
            Strides of the convolutional layers.

        padding : str
            Padding of the block.

        patch_size : tuple
            Patch size of the Unfold layer.

        num_transformer_layers : int
            Number of Transformer layers.

        activation : str, optional
            Activation function of the block.
                The options are:
                    "relu"
                        Rectified Linear Unit activation function.
                    "leaky_relu"
                        Leaky Rectified Linear Unit activation function.
                    self.activation
                        Hard Swish activation function.
                    "mish"
                        Mish activation function.

        num_transformer_heads : int, optional
            Number of heads for each Transformer layer.

        dropout_transformer : float, optional
            Fraction of the input units to drop in the Transformer layers. It must be between 0 and 1.

        aproximator_type : str, optional
            Type of MLP aproximator.
                The options are:
                    "vanilla"
                        Vanilla MLP.
                    "siren"
                        SIREN MLP.

        dropout_mlp : float, optional
            Fraction of the input units to drop in the MLP. It must be between 0 and 1.
            Note that in the original implementation of the SIREN, the dropout rate is 0.0.


        Returns
        -------
        None.
        
        """

        if not isinstance(projection_dim, int):
            raise TypeError("projection_dim must be an integer")
        if projection_dim < 0:
            raise ValueError("projection_dim must be positive")
        if not isinstance(kernel_size, list):
            raise TypeError("kernel_size must be a list")
        if len(kernel_size) != 2:
            raise ValueError("kernel_size must contain 2 tuples")
        if not all(isinstance(k, tuple) for k in kernel_size):
            raise TypeError("kernel_size must contain only tuples")
        if not all([len(k) == 3 for k in kernel_size]):
            raise ValueError("kernel_size must contain only tuples of length 3")
        for k in kernel_size:
            if not all(isinstance(i, int) for i in k):
                raise TypeError("kernel_size elements must contain only integers")
            if not all([i > 0 for i in k]):
                raise ValueError("kernel_size elements must contain only positive integers")
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
        if not isinstance(patch_size, tuple):
            raise TypeError("patch_size must be a tuple")
        if len(patch_size) != 3:
            raise ValueError("patch_size must be a tuple of length 3")
        if not all([isinstance(p, int) for p in patch_size]):
            raise TypeError("patch_size must contain only integers")
        if not all([p > 0 for p in patch_size]):
            raise ValueError("patch_size must contain only positive integers")
        if not isinstance(num_transformer_layers, int):
            raise TypeError("num_transformer_layers must be an integer")
        if num_transformer_layers < 0:
            raise ValueError("num_transformer_layers must be positive")
        if not isinstance(activation, str):
            raise TypeError("activation must be a string")
        if activation not in ["relu", "leaky_relu", "hard_swish", "mish"]:
            raise ValueError("Unknown activation function")
        if not isinstance(num_transformer_heads, int):
            raise TypeError("num_transformer_heads must be an integer")
        if num_transformer_heads < 0:
            raise ValueError("num_transformer_heads must be positive")
        if not isinstance(dropout_mha, float):
            raise TypeError("dropout_mha rate must be a float")
        if dropout_mha < 0 or dropout_mha > 1:
            raise ValueError("dropout_mha rate must be between 0 and 1")
        if not isinstance(aproximator_type, str):
            raise TypeError("aproximator_type must be a string")
        if aproximator_type not in ["vanilla", "siren"]:
            raise ValueError("Unknown aproximator type")
        if not isinstance(dropout_mlp, float):
            raise TypeError("dropout_mlp rate must be a float")
        if dropout_mlp < 0 or dropout_mlp > 1:
            raise ValueError("dropout_mlp rate must be between 0 and 1")


        super().__init__(**kwargs)

        self.projection_dim = projection_dim
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.patch_size = patch_size
        self.num_transformer_layers = num_transformer_layers
        self.num_transformer_heads = num_transformer_heads
        self.dropout_mha = dropout_mha
        self.aproximator_type = aproximator_type
        self.dropout_mlp = dropout_mlp


    def build(self, input_shape: tf.TensorShape) -> None:

        """

        Build method of the MobileViViT block.


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

        # Local Features
        self.local_feature_extract = ConvLayer(filters=self.projection_dim, kernel_size=self.kernel_size[0], 
                                               strides=self.strides, padding=self.padding, batch_norm=False, 
                                               activation=self.activation)
        self.local_feature_extract_1 = ConvLayer(filters=self.projection_dim, kernel_size=(1, 1, 1), 
                                                 strides=(1, 1, 1), padding="valid", batch_norm=False, 
                                                 activation=self.activation)

        # Global Features
        self.unfold = Unfold(embedding_dim=self.projection_dim, patch_size=self.patch_size, padding=self.padding)
        self.transformer = Transformer(projection_dim=self.projection_dim, 
                                       num_transformer_layers=self.num_transformer_layers, 
                                       num_transformer_heads=self.num_transformer_heads, 
                                       dropout_mha=self.dropout_mha, 
                                       aproximator_type=self.aproximator_type, 
                                       dropout_mlp=self.dropout_mlp)
        temp_shape = self.local_feature_extract_1.compute_output_shape(self.local_feature_extract
                                                 .compute_output_shape(input_shape))
        self.fold = Fold(embedding_dim=self.projection_dim,
                         shape=(temp_shape[0],
                                int(temp_shape[1]/self.patch_size[0]), 
                                int(temp_shape[2]/self.patch_size[1]), 
                                int(temp_shape[3]/self.patch_size[2]), 
                                self.projection_dim))

        # Fusion
        if self.patch_size != (1, 1, 1) and self.strides != (1, 1, 1):
            self.unify = tf.keras.layers.AveragePooling3D(pool_size=self.strides, 
                                                          strides=self.strides, 
                                                          padding=self.padding)
            self.unify_1 = tf.keras.layers.AveragePooling3D(pool_size=self.patch_size, 
                                                          strides=self.patch_size, 
                                                          padding="valid")
        elif self.patch_size != (1, 1, 1):
            self.unify = tf.keras.layers.AveragePooling3D(pool_size=self.patch_size, 
                                                          strides=self.patch_size, 
                                                          padding="valid")
            
        self.connect = ConvLayer(filters=input_shape[-1], kernel_size=(1, 1, 1), 
                                 strides=(1, 1, 1), padding="valid", 
                                 batch_norm=False, activation=self.activation)
        self.concat = tf.keras.layers.Concatenate()
        self.fuse = ConvLayer(filters=self.projection_dim, kernel_size=self.kernel_size[1], 
                              strides=self.strides, padding=self.padding, batch_norm=False, 
                              activation=self.activation)


    def call(self, X: tf.Tensor) -> tf.Tensor:

        """

        Call method of the MobileViViT block.


        Parameters
        ----------
        X : tf.Tensor
            Input tensor.
        
            
        Returns
        -------
        X_transformed : tf.Tensor
            Output tensor.

        """

        local_features = self.local_feature_extract_1(self.local_feature_extract(X))

        folded_global_features = self.fold(self.transformer(self.unfold(local_features)))

        if self.patch_size != (1, 1, 1) and self.strides != (1, 1, 1):
            X = self.unify_1(self.unify(X))
        elif self.patch_size != (1, 1, 1):
            X = self.unify(X)


        return self.fuse(self.concat([X, self.connect(folded_global_features)]))


    def get_config(self) -> dict[str, Any]:

        """

        Method to get the configuration of the MobileViViT block.


        Parameters
        ----------
        None.


        Returns
        -------
        config : dict
            Configuration of the MobileViViT block.

        """

        config = super().get_config()

        config.update({
            "projection_dim": self.projection_dim,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "activation": self.activation,
            "patch_size": self.patch_size,
            "num_transformer_layers": self.num_transformer_layers,
            "num_transformer_heads": self.num_transformer_heads,
            "dropout_mha": self.dropout_mha,
            "aproximator_type": self.aproximator_type,
            "dropout_mlp": self.dropout_mlp
        })


        return config
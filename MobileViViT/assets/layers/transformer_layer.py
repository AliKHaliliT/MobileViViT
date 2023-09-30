import tensorflow as tf
from ..blocks.mlp import MLP
from ..blocks.siren import SIREN
from typing import Any


class TransformerLayer(tf.keras.layers.Layer):

    """
    
    Slightly modified implementation of the transformer layer (Encoder Only) based 
    on https://keras.io/examples/vision/image_classification_with_vision_transformer/
    
    """

    def __init__(self, projection_dim: int, num_transformer_heads: int, 
                 dropout_mha: float = 0.1, aproximator_type: str = "vanilla", 
                 activation: str = "hard_swish", dropout_mlp: float = 0.1, **kwargs) -> None:
        
        """
        
        Constructor of the Transformer layer.


        Parameters
        ----------
        projection_dim : int
            Dimension of the projection space.
        
        num_transformer_heads : int
            Number of attention heads.

        dropout_mha : float, optional
            Fraction of the input units to drop in the MultiHeadAttention layer. It must be between 0 and 1. The default is 0.1.

        aproximator_type : str, optional
            Type of the MLP aproximator.
                The options are:
                    "vanilla"
                        Vanilla MLP.
                    "siren"
                        SIREN MLP.

        activation : str, optional
            Activation function of the MLP. Only used if aproximator_type is "vanilla". The default is "hard_swish".
                The options are:
                    "relu"
                        Rectified Linear Unit activation function.
                    "leaky_relu"
                        Leaky Rectified Linear Unit activation function.
                    "hard_swish"
                        Hard Swish activation function.
                    "mish"
                        Mish activation function.

        dropout_mlp : float, optional
            Fraction of the input units to drop in the MLP. It must be between 0 and 1. The default is 0.1.
            Note that in the original implementation of the SIREN, the dropout rate is 0.0.


        Returns
        -------
        None.

        """

        if not isinstance(projection_dim, int):
            raise TypeError("projection_dim must be an integer")
        if projection_dim < 0:
            raise ValueError("projection_dim must be positive")
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
        if not isinstance(activation, str):
            raise TypeError("activation must be a string")
        if activation not in ["relu", "leaky_relu", "hard_swish", "mish"]:
            raise ValueError("Unknown activation function")
        if not isinstance(dropout_mlp, float):
            raise TypeError("dropout_mlp rate must be a float")
        if dropout_mlp < 0 or dropout_mlp > 1:
            raise ValueError("dropout_mlp rate must be between 0 and 1")


        super().__init__(**kwargs)

        self.projection_dim = projection_dim
        self.num_transformer_heads = num_transformer_heads
        self.dropout_mha = dropout_mha
        self.aproximator_type = aproximator_type
        self.activation = activation
        self.dropout_mlp = dropout_mlp


    def build(self, input_shape: tf.TensorShape) -> None:

        """

        Build method of the Transformer layer.


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

        self.layer_normalize = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=self.num_transformer_heads, 
                                                            key_dim=self.projection_dim, 
                                                            dropout=self.dropout_mha)
        self.add = tf.keras.layers.Add()

        self.layer_normalize_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        if self.aproximator_type == "vanilla":
            self.mlp = MLP([input_shape[-1], input_shape[-1]], activation=self.activation, dropout=self.dropout_mlp)
        elif self.aproximator_type == "siren":
            self.mlp = SIREN([input_shape[-1], input_shape[-1]], dropout=self.dropout_mlp)
        self.add_1 = tf.keras.layers.Add()
        

    def call(self, X: tf.Tensor) -> tf.Tensor:

        """

        Call method of the Transformer layer.


        Parameters
        ----------
        X : tf.Tensor
            Input tensor.
        
            
        Returns
        -------
        X_transformed : tf.Tensor
            Output tensor.

        """

        X_normalized = self.layer_normalize(X)

        X_attentioned = self.add([X_normalized, self.attention(X_normalized, X_normalized)])


        return self.add_1([X_attentioned, self.mlp(self.layer_normalize_1(X_attentioned))])

    

    def get_config(self) -> dict[str, Any]:

        """

        Method to get the configuration of the Transformer layer.


        Parameters
        ----------
        None.


        Returns
        -------
        config : dict
            Configuration of the Transformer layer.

        """
            
        config = super().get_config()

        config.update({
            "projection_dim": self.projection_dim,
            "num_transformer_heads": self.num_transformer_heads,
            "dropout_mha": self.dropout_mha,
            "aproximator_type": self.aproximator_type,
            "activation": self.activation,
            "dropout_mlp": self.dropout_mlp
        })


        return config
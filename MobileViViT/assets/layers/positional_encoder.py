import tensorflow as tf
from typing import Any


class PositionalEncoder(tf.keras.layers.Layer):

    """
    
    Positional Encoder Layer from the paper "ViViT: A Video Vision Transformer"
    https://arxiv.org/abs/2103.15691
    
    """

    def __init__(self, embedding_dim: int, **kwargs) -> None:

        """
        
        Constructor of the PositionalEncoder layer.

        
        Parameters
        ----------
        embedding_dim : int
            Dimension of the embedding vector.


        Returns
        -------
        None.
        
        """

        if not isinstance(embedding_dim, int):
            raise TypeError("embedding_dim must be an integer")
        if embedding_dim < 0:
            raise ValueError("embedding_dim must be positive")
        

        super().__init__(**kwargs)

        self.embedding_dim = embedding_dim


    def build(self, input_shape: tf.TensorShape) -> None:

        """

        Build method of the PositionalEncoder layer.


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
        
        num_tokens = input_shape[1]
        self.position_embedding = tf.keras.layers.Embedding(input_dim=num_tokens, output_dim=self.embedding_dim)
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)


    def call(self, X: tf.Tensor) -> tf.Tensor:

        """

        Call method of the PositionalEncoder layer.


        Parameters
        ----------
        X : tf.Tensor
            Input tensor.
        
            
        Returns
        -------
        X_transformed : tf.Tensor
            Output tensor.

        """


        return tf.add(X, self.position_embedding(self.positions))


    def get_config(self) -> dict[str, Any]:

        """

        Method to get the configuration of the PositionalEncoder layer.


        Parameters
        ----------
        None.


        Returns
        -------
        config : dict
            Configuration of the PositionalEncoder layer.

        """
            
        config = super().get_config()

        config.update({
            "embedding_dim": self.embedding_dim
        })


        return config
import tensorflow as tf
from typing import Any


class Fold(tf.keras.layers.Layer):

    """
    
    Fold layer adapted for higher dimensional tasks from 
    the paper "MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer"
    Link: https://arxiv.org/abs/2110.02178
    
    """

    def __init__(self, embedding_dim: int, shape: tuple[int, int, int, int, int], **kwargs):

        """
        
        Constructor of the Fold layer.


        Parameters
        ----------
        embedding_dim : int
            Dimension of the embedding.

        shape : tuple
            Shape of the output tensor. It must be equal to the shape of the input tensor of the Unfold layer.


        Returns
        -------
        None.
        
        """

        if not isinstance(embedding_dim, int):
            raise TypeError("embedding_dim must be an integer")
        if embedding_dim < 0:
            raise ValueError("embedding_dim must be positive")
        if not isinstance(shape, tuple) and not isinstance(shape, tf.TensorShape):
            raise TypeError("shape must be either a tuple or a TensorShape object")
        if len(shape) != 5:
            raise ValueError("shape must contain 5 dimensions")
        if not all(dim > 0 for dim in shape[1:]):
            raise ValueError("shape must contain only positive integers")
        

        super().__init__(**kwargs)

        self.embedding_dim = embedding_dim
        self.shape = shape


    def build(self, input_shape: tf.TensorShape) -> None:

        """

        Build method of the Fold layer.


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

        self.fold = tf.keras.layers.Reshape((self.shape[1], self.shape[2], self.shape[3], self.embedding_dim))


    def call(self, X: tf.Tensor) -> tf.Tensor:

        """

        Call method of the Fold layer.


        Parameters
        ----------
        X : tf.Tensor
            Input tensor.

        
        Returns
        -------
        X_transformed : tf.Tensor
            Output tensor.

        """
    

        return self.fold(X)

    
    def get_config(self) -> dict[str, Any]:

        """

        Method to get the configuration of the Fold layer.


        Parameters
        ----------
        None.


        Returns
        -------
        config : dict
            Configuration of the Fold layer.

        """
            
        config = super().get_config()

        config.update({
            "embedding_dim": self.embedding_dim,
            "shape": self.shape
        })


        return config
import tensorflow as tf
from .tubelet_embedding import TubeletEmbedding
from .positional_encoder import PositionalEncoder
from typing import Any


class Unfold(tf.keras.layers.Layer):

    """

    Slightly modified implementation of the Unfold layer adapted for higher dimensional tasks from 
    the paper "MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer"
    https://arxiv.org/abs/2110.02178

    """


    def __init__(self, embedding_dim: int, patch_size: tuple[int, int, int], 
                 padding: str, **kwargs) -> None:

        """

        Constructor of the Unfold layer.


        Parameters
        ----------
        embedding_dim : int
            Dimension of the embedding vector.
        
        patch_size : tuple
            Size of the patchs to be extracted. 
            For more information, check the 
            "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" 
            and,
            "ViViT: A Video Vision Transformer" papers.
            Link: https://arxiv.org/abs/2010.11929 and https://arxiv.org/abs/2103.15691

        padding : str
            Padding of the layer. 
                The options are:
                    "valid"
                        No padding.
                    "same"
                        Padding with zeros.

        
        Returns
        -------
        None.

        """

        if not isinstance(embedding_dim, int):
            raise TypeError("embedding_dim must be an integer")
        if embedding_dim < 0:
            raise ValueError("embedding_dim must be positive")
        if not isinstance(patch_size, tuple):
            raise TypeError("patch_size must be a tuple")
        if len(patch_size) != 3:
            raise ValueError("patch_size must be a tuple of length 3")
        if not all([isinstance(p, int) for p in patch_size]):
            raise TypeError("patch_size must contain only integers")
        if not all([p > 0 for p in patch_size]):
            raise ValueError("patch_size must contain only positive integers")
        if not isinstance(padding, str):
            raise TypeError("padding must be a string")
        if padding not in ["valid", "same"]:
            raise ValueError("padding must be either valid or same")


        super().__init__(**kwargs)

        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.padding = padding


    def build(self, input_shape: tf.TensorShape) -> None:

        """

        Build method of the Unfold layer.


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

        self.embed = TubeletEmbedding(embedding_dim=self.embedding_dim, patch_size=self.patch_size, padding=self.padding)
        self.encode = PositionalEncoder(embedding_dim=self.embedding_dim)
        

    def call(self, X: tf.Tensor) -> tf.Tensor:

        """

        Call method of the Unfold layer.


        Parameters
        ----------
        X : tf.Tensor
            Input tensor.
        
            
        Returns
        -------
        X_transformed : tf.Tensor
            Output tensor.

        """


        return self.encode(self.embed(X))


    def get_config(self) -> dict[str, Any]:

        """

        Method to get the configuration of the Unfold layer.


        Parameters
        ----------
        None.


        Returns
        -------
        config : dict
            Configuration of the Unfold layer.

        """
            
        config = super().get_config()

        config.update({
            "embedding_dim": self.embedding_dim,
            "patch_size": self.patch_size,
            "padding": self.padding
        })


        return config
import tensorflow as tf
from .assets.layers.conv_layer import ConvLayer
from .assets.blocks.mvnblock import MVNBlock
from .assets.blocks.mobilevivit import MobileViViT


class MobileViViTXS(tf.keras.Model):

    """
    
    MobileViViT-XS Variant Architecture
    The architecture is based on the MobileViT-XS variant from 
    the paper "MobileViT: Light-weight, General-purpose, and Mobile-friendly",
    adapted for higher dimensional tasks.
    Link: https://arxiv.org/abs/2110.02178

    The input size for this variant is 50 x 256 x 256 x 3. The 50 here refers to the 
    the total number of frames. Each video is considered to be in 5 frames per second (fps).
    
    Though the model can work with any input size, it is recommended to use the input size mentioned above.

    Note that in this adaptation, the spatial properties of the base architecture 
    are tried to be preserved as much as possible. 

    Also note that the final output of the model is not passed through an activation function.

    Given the parameters in the paper, the model has around 12 million parameters.
    
    """

    def __init__(self, num_output_units) -> None:

        """

        Constructor of the MobileViViT-XS model.
        
        
        Parameters
        ----------
        num_output_units : int
            Number of output units.
        
        
        Returns
        -------
        None.
        
        """

        if not isinstance(num_output_units, int):
            raise TypeError("num_output_units must be an integer")
        if num_output_units < 0:
            raise ValueError("num_output_units must be positive")


        super().__init__()

        expansion_factor = 4

        # Stem
        ## Stage 0
        self.conv_layer = ConvLayer(filters=16, kernel_size=(5, 3, 3), strides=(2, 2, 2), 
                                    padding="same", batch_norm=False, activation="hard_swish")
        self.mvnblock = MVNBlock(expansion_filters=32 * expansion_factor, filters=16, 
                                 kernel_size=(5, 3, 3), strides=(1, 1, 1), 
                                 padding="same", activation="hard_swish")
        
        # Backbone
        ## Stage 1
        self.mvnblock_1 = MVNBlock(expansion_filters=48 * expansion_factor, filters=24, 
                                   kernel_size=(3, 3, 3), strides=(2, 2, 2), 
                                   padding="same", activation="hard_swish")
        self.mvnblock_2 = MVNBlock(expansion_filters=48 * expansion_factor, filters=24,
                                   kernel_size=(3, 3, 3), strides=(1, 1, 1), 
                                   padding="same", activation="hard_swish")
        self.mvnblock_3 = MVNBlock(expansion_filters=48 * expansion_factor, filters=24,
                                   kernel_size=(3, 3, 3), strides=(1, 1, 1), 
                                   padding="same", activation="hard_swish")

        ## Stage 2
        self.mvnblock_4 = MVNBlock(expansion_filters=64 * expansion_factor, filters=48,
                                   kernel_size=(3, 3, 3), strides=(2, 2, 2), 
                                   padding="same", activation="hard_swish")
        self.mobilevivit = MobileViViT(projection_dim=96, kernel_size=[(3, 3, 3), (3, 3, 3)],
                                       strides=(1, 1, 1), padding="same", patch_size=(1, 2, 2), 
                                       num_transformer_layers=2)
        
        ## Stage 3
        self.mvnblock_5 = MVNBlock(expansion_filters=80 * expansion_factor, filters=64,
                                   kernel_size=(2, 3, 3), strides=(1, 1, 1), 
                                   padding="same", activation="hard_swish")
        self.mobilevivit_1 = MobileViViT(projection_dim=120, kernel_size=[(2, 3, 3), (2, 3, 3)],
                                         strides=(1, 1, 1), padding="same", patch_size=(1, 2, 2), 
                                         num_transformer_layers=4)
        
        ## Stage 4
        self.mvnblock_6 = MVNBlock(expansion_filters=96 * expansion_factor, filters=80,
                                   kernel_size=(1, 3, 3), strides=(1, 1, 1), 
                                   padding="same", activation="hard_swish")
        self.mobilevivit_2 = MobileViViT(projection_dim=144, kernel_size=[(1, 3, 3), (1, 3, 3)],
                                         strides=(1, 1, 1), padding="same", patch_size=(1, 2, 2), 
                                         num_transformer_layers=3)
        self.conv_layer_1 = ConvLayer(filters=384, kernel_size=(1, 1, 1), strides=(1, 1, 1),
                                        padding="same", batch_norm=False, activation="hard_swish")
        
        # Head
        self.global_average_pool = tf.keras.layers.GlobalAveragePooling3D()
        self.dense = tf.keras.layers.Dense(units=num_output_units)


    def call(self, X) -> tf.Tensor:

        """

        Call method of the MobileViViT-XS model.
        
        
        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        
        
        Returns
        -------
        tf.Tensor
            Output tensor.
        
        """

        X = self.conv_layer(X)
        X = self.mvnblock(X)
        
        X = self.mvnblock_1(X)
        X = self.mvnblock_2(X)
        X = self.mvnblock_3(X)

        X = self.mvnblock_4(X)
        X = self.mobilevivit(X)
        
        X = self.mvnblock_5(X)
        X = self.mobilevivit_1(X)
        
        X = self.mvnblock_6(X)
        X = self.mobilevivit_2(X)
        X = self.conv_layer_1(X)
        
        X = self.global_average_pool(X)
        X = self.dense(X)


        return X     
import unittest
from MobileViViT.assets.blocks.mobilevivit import MobileViViT
import tensorflow as tf


class TestMobileViViT(unittest.TestCase):

    def test_projection__dim_wrong__type_type__error(self):

        # Arrange
        projection_dim = None

        # Act and Assert
        with self.assertRaises(TypeError):
            MobileViViT(projection_dim=projection_dim, kernel_size=[(1, 1, 1), (1, 1, 1)], 
                        strides=(1, 1, 1), padding="same", patch_size=(1, 1, 1), 
                        num_transformer_layers=2)


    def test_projection__dim_wrong__value_value__error(self):
        
        # Arrange
        projection_dim = -1

        # Act and Assert
        with self.assertRaises(ValueError):
            MobileViViT(projection_dim=projection_dim, kernel_size=[(1, 1, 1), (1, 1, 1)], 
                        strides=(1, 1, 1), padding="same", patch_size=(1, 1, 1), 
                        num_transformer_layers=2)

    def test_kernel__size_wrong__type_type__error(self):

        # Arrange
        kernel_size = None

        # Act and Assert
        with self.assertRaises(TypeError):
            MobileViViT(projection_dim=1, kernel_size=kernel_size, 
                        strides=(1, 1, 1), padding="same", patch_size=(1, 1, 1), 
                        num_transformer_layers=2)


    def test_kernel__size_not__len__two_value__error(self):

        # Arrange
        kernel_size = [(1, 1, 1)]

        # Act and Assert
        with self.assertRaises(ValueError):
            MobileViViT(projection_dim=1, kernel_size=kernel_size, 
                        strides=(1, 1, 1), padding="same", patch_size=(1, 1, 1), 
                        num_transformer_layers=2)


    def test_kernel__size_wrong__type__instances_type__error(self):

        # Arrange
        kernel_size = [(1, 1, 1), None]

        # Act and Assert
        with self.assertRaises(TypeError):
            MobileViViT(projection_dim=1, kernel_size=kernel_size, 
                        strides=(1, 1, 1), padding="same", patch_size=(1, 1, 1), 
                        num_transformer_layers=2)
            

    def test_kernel__size_kernel__size__elements__not__len__three_value__error(self):

        # Arrange
        kernel_size = [(1, 1, 1), (1, 1)]

        # Act and Assert
        with self.assertRaises(ValueError):
            MobileViViT(projection_dim=1, kernel_size=kernel_size, 
                        strides=(1, 1, 1), padding="same", patch_size=(1, 1, 1), 
                        num_transformer_layers=2)


    def test_kernel__size_wrong__kernel__size__elements__type__instances_type__error(self):

        # Arrange
        kernel_size = [(1, 1, 1), (1, 1, None)]

        # Act and Assert
        with self.assertRaises(TypeError):
            MobileViViT(projection_dim=1, kernel_size=kernel_size, 
                        strides=(1, 1, 1), padding="same", patch_size=(1, 1, 1), 
                        num_transformer_layers=2)
            

    def test_kernel__size_wrong__value_value__error(self):

        # Arrange
        kernel_size = [(-1, -1, -1), (-1, -1, -1)]

        # Act and Assert
        with self.assertRaises(ValueError):
            MobileViViT(projection_dim=1, kernel_size=kernel_size, 
                        strides=(1, 1, 1), padding="same", patch_size=(1, 1, 1), 
                        num_transformer_layers=2)

    
    def test_strides_wrong__type_type__error(self):

        # Arrange
        strides = None

        # Act and Assert
        with self.assertRaises(TypeError):
            MobileViViT(projection_dim=1, kernel_size=[(1, 1, 1), (1, 1, 1)], 
                        strides=strides, padding="same", patch_size=(1, 1, 1), 
                        num_transformer_layers=2)

    
    def test_strides_not__len__three_value__error(self):

        # Arrange
        strides = (1, 1)

        # Act and Assert
        with self.assertRaises(ValueError):
            MobileViViT(projection_dim=1, kernel_size=[(1, 1, 1), (1, 1, 1)], 
                        strides=strides, padding="same", patch_size=(1, 1, 1), 
                        num_transformer_layers=2)


    def test_strides_wrong__type__instances_type__error(self):

        # Arrange
        strides = (1, 1, None)

        # Act and Assert
        with self.assertRaises(TypeError):
            MobileViViT(projection_dim=1, kernel_size=[(1, 1, 1), (1, 1, 1)], 
                        strides=strides, padding="same", patch_size=(1, 1, 1), 
                        num_transformer_layers=2)

    
    def test_strides_wrong__value_value__error(self):
        
        # Arrange
        strides = (-1, -1, -1)

        # Act and Assert
        with self.assertRaises(ValueError):
            MobileViViT(projection_dim=1, kernel_size=[(1, 1, 1), (1, 1, 1)], 
                        strides=strides, padding="same", patch_size=(1, 1, 1), 
                        num_transformer_layers=2)


    def test_padding_wrong__type_type__error(self):

        # Arrange
        padding = None

        # Act and Assert
        with self.assertRaises(TypeError):
            MobileViViT(projection_dim=1, kernel_size=[(1, 1, 1), (1, 1, 1)], 
                        strides=(1, 1, 1), padding=padding, patch_size=(1, 1, 1), 
                        num_transformer_layers=2)

    
    def test_padding_wrong__value_value__error(self):
        
        # Arrange
        padding = "test"

        # Act and Assert
        with self.assertRaises(ValueError):
            MobileViViT(projection_dim=1, kernel_size=[(1, 1, 1), (1, 1, 1)], 
                        strides=(1, 1, 1), padding=padding, patch_size=(1, 1, 1),
                        num_transformer_layers=2)
            

    def test_patch__size_wrong__type_type__error(self):

        # Arrange
        patch_size = None

        # Act and Assert
        with self.assertRaises(TypeError):
            MobileViViT(projection_dim=1, kernel_size=[(1, 1, 1), (1, 1, 1)],
                        strides=(1, 1, 1), padding="same", patch_size=patch_size, 
                        num_transformer_layers=2)


    def test_patch__size_not__len__three_value__error(self):

        # Arrange
        patch_size = (1, 1)

        # Act and Assert
        with self.assertRaises(ValueError):
            MobileViViT(projection_dim=1, kernel_size=[(1, 1, 1), (1, 1, 1)], 
                        strides=(1, 1, 1), padding="same", patch_size=patch_size, 
                        num_transformer_layers=2)


    def test_patch__size_wrong__type__instances_type__error(self):

        # Arrange
        patch_size = (1, 1, None)

        # Act and Assert
        with self.assertRaises(TypeError):
            MobileViViT(projection_dim=1, kernel_size=[(1, 1, 1), (1, 1, 1)], 
                        strides=(1, 1, 1), padding="same", patch_size=patch_size, 
                        num_transformer_layers=2)


    def test_patch__size_wrong__value_value__error(self):

        # Arrange
        patch_size = (-1, -1, -1)

        # Act and Assert
        with self.assertRaises(ValueError):
            MobileViViT(projection_dim=1, kernel_size=[(1, 1, 1), (1, 1, 1)], 
                        strides=(1, 1, 1), padding="same", patch_size=patch_size, 
                        num_transformer_layers=2)


    def test_num__transformer__layers_wrong__type_type__error(self):

        # Arrange
        num_transformer_layers = None

        # Act and Assert
        with self.assertRaises(TypeError):
            MobileViViT(projection_dim=1, kernel_size=[(1, 1, 1), (1, 1, 1)], 
                        strides=(1, 1, 1), padding="same", patch_size=(1, 1, 1), 
                        num_transformer_layers=num_transformer_layers)


    def test_num__transformer__layers_wrong__value_value__error(self):

        # Arrange
        num_transformer_layers = -1

        # Act and Assert
        with self.assertRaises(ValueError):
            MobileViViT(projection_dim=1, kernel_size=[(1, 1, 1), (1, 1, 1)], 
                        strides=(1, 1, 1), padding="same", patch_size=(1, 1, 1),
                        num_transformer_layers=num_transformer_layers)
            

    def test_activation_wrong__type_type__error(self):

        # Arrange
        activation = None

        # Act and Assert
        with self.assertRaises(TypeError):
            MobileViViT(projection_dim=1, kernel_size=[(1, 1, 1), (1, 1, 1)], 
                        strides=(1, 1, 1), padding="same", patch_size=(1, 1, 1), 
                        num_transformer_layers=2, activation=activation)

        
    def test_activation_wrong__value_value__error(self):

        # Arrange
        activation = "test"

        # Act and Assert
        with self.assertRaises(ValueError):
            MobileViViT(projection_dim=1, kernel_size=[(1, 1, 1), (1, 1, 1)], 
                        strides=(1, 1, 1), padding="same", patch_size=(1, 1, 1), 
                        num_transformer_layers=2, activation=activation)

    
    def test_num__transformer__heads_wrong__type_type__error(self):
        
        # Arrange
        num_transformer_heads = None

        # Act and Assert
        with self.assertRaises(TypeError):
            MobileViViT(projection_dim=1, kernel_size=[(1, 1, 1), (1, 1, 1)], 
                        strides=(1, 1, 1), padding="same", patch_size=(1, 1, 1), 
                        num_transformer_layers=2, num_transformer_heads=num_transformer_heads)

    
    def test_num__transformer__heads_wrong__value_value__error(self):

        # Arrange
        num_transformer_heads = -1

        # Act and Assert
        with self.assertRaises(ValueError):
            MobileViViT(projection_dim=1, kernel_size=[(1, 1, 1), (1, 1, 1)], 
                        strides=(1, 1, 1), padding="same", patch_size=(1, 1, 1), 
                        num_transformer_layers=2, num_transformer_heads=num_transformer_heads)


    def test_dropout__mha_wrong__type_type__error(self):

        # Arrange
        dropout_mha = "test"

        # Act and Assert
        with self.assertRaises(TypeError):
            MobileViViT(projection_dim=1, kernel_size=[(1, 1, 1), (1, 1, 1)], 
                        strides=(1, 1, 1), padding="same", patch_size=(1, 1, 1), 
                        num_transformer_layers=2, dropout_mha=dropout_mha)


    def test_dropout__mha_wrong__value_value__error(self):

        # Arrange
        dropout_mha = 1.2

        # Act and Assert
        with self.assertRaises(ValueError):
            MobileViViT(projection_dim=1, kernel_size=[(1, 1, 1), (1, 1, 1)], 
                        strides=(1, 1, 1), padding="same", patch_size=(1, 1, 1), 
                        num_transformer_layers=2, dropout_mha=dropout_mha)


    def test_aproximator__type_wrong__type_type__error(self):
        
        # Arrange
        aproximator_type = None

        # Act and Assert
        with self.assertRaises(TypeError):
            MobileViViT(projection_dim=1, kernel_size=[(1, 1, 1), (1, 1, 1)], 
                        strides=(1, 1, 1), padding="same", patch_size=(1, 1, 1), 
                        num_transformer_layers=2, aproximator_type=aproximator_type)


    def test_aproximator__type_wrong__value_value__error(self):

        # Arrange
        aproximator_type = "test"

        # Act and Assert
        with self.assertRaises(ValueError):
            MobileViViT(projection_dim=1, kernel_size=[(1, 1, 1), (1, 1, 1)], 
                        strides=(1, 1, 1), padding="same", patch_size=(1, 1, 1), 
                        num_transformer_layers=2, aproximator_type=aproximator_type)


    def test_dropout__mlp_wrong__type_type__error(self):

        # Arrange
        dropout_mlp = "test"

        # Act and Assert
        with self.assertRaises(TypeError):
            MobileViViT(projection_dim=1, kernel_size=[(1, 1, 1), (1, 1, 1)], 
                        strides=(1, 1, 1), padding="same", patch_size=(1, 1, 1), 
                        num_transformer_layers=2, dropout_mlp=dropout_mlp)


    def test_dropout__mlp_wrong__value_value__error(self):

        # Arrange
        dropout_mlp = 1.2

        # Act and Assert
        with self.assertRaises(ValueError):
            MobileViViT(projection_dim=1, kernel_size=[(1, 1, 1), (1, 1, 1)], 
                        strides=(1, 1, 1), padding="same", patch_size=(1, 1, 1), 
                        num_transformer_layers=2, dropout_mlp=dropout_mlp)


    def test_input__error_not__tensor_type__error(self):

        # Arrange
        input_tensor = [[[[[1, 2, 3]]]]]

        # Act and Assert
        with self.assertRaises(TypeError):
            MobileViViT(projection_dim=1, kernel_size=[(1, 1, 1), (1, 1, 1)], 
                        strides=(1, 1, 1), padding="same", patch_size=(1, 1, 1), 
                        num_transformer_layers=2)(input_tensor)


    def test_output_tensor_intended__shape(self):

        # Arrange
        input_tensor = tf.random.normal((1, 1, 1, 1, 3))

        # Act
        output = MobileViViT(projection_dim=1, kernel_size=[(1, 1, 1), (1, 1, 1)], 
                             strides=(1, 1, 1), padding="same", patch_size=(1, 1, 1), 
                             num_transformer_layers=2)(input_tensor)

        # Assert
        self.assertEqual(output.shape, (1, 1, 1, 1, 1))


    def test_get__config_layer_non__empty__dict(self):

        # Arrange and Act
        config = MobileViViT(projection_dim=1, kernel_size=[(1, 1, 1), (1, 1, 1)], 
                             strides=(1, 1, 1), padding="same", patch_size=(1, 1, 1), 
                             num_transformer_layers=2).get_config()

        # Assert
        self.assertGreater(len(config), 0)


    def test_from__config_layer__config__dict_input__config__dict(self):

        # Arrange
        config = MobileViViT(projection_dim=1, kernel_size=[(1, 1, 1), (1, 1, 1)], 
                             strides=(1, 1, 1), padding="same", patch_size=(1, 1, 1), 
                             num_transformer_layers=2).get_config()

        # Act
        cloned_layer = MobileViViT(projection_dim=1, kernel_size=[(1, 1, 1), (1, 1, 1)], 
                                   strides=(1, 1, 1), padding="same", patch_size=(1, 1, 1),
                                   num_transformer_layers=2).from_config(config)

        # Assert
        self.assertEqual(config, cloned_layer.get_config())


if __name__ == "__main__":
    unittest.main()
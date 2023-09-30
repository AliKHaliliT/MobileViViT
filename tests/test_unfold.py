import unittest
from MobileViViT.assets.layers.unfold import Unfold
import tensorflow as tf


class TestUnfold(unittest.TestCase):

    def test_embedding__dim_wrong__type_type__error(self):

        # Arrange
        embedding_dim = None

        # Act and Assert
        with self.assertRaises(TypeError):
            Unfold(embedding_dim=embedding_dim, patch_size=(1, 1, 1), padding="same")


    def test_embedding__dim_wrong__value_value__error(self):
        
        # Arrange
        embedding_dim = -1

        # Act and Assert
        with self.assertRaises(ValueError):
            Unfold(embedding_dim=embedding_dim, patch_size=(1, 1, 1), padding="same")


    def test_patch__size_wrong__type_type__error(self):

        # Arrange
        patch_size = None

        # Act and Assert
        with self.assertRaises(TypeError):
            Unfold(embedding_dim=1, patch_size=patch_size, padding="same")


    def test_patch__size_not__len__three_value__error(self):

        # Arrange
        patch_size = (1, 1)

        # Act and Assert
        with self.assertRaises(ValueError):
            Unfold(embedding_dim=1, patch_size=patch_size, padding="same")


    def test_patch__size_wrong__type__instances_type__error(self):

        # Arrange
        patch_size = (1, 1, None)

        # Act and Assert
        with self.assertRaises(TypeError):
            Unfold(embedding_dim=1, patch_size=patch_size, padding="same")


    def test_patch__size_wrong__value_value__error(self):

        # Arrange
        patch_size = (-1, -1, -1)

        # Act and Assert
        with self.assertRaises(ValueError):
            Unfold(embedding_dim=1, patch_size=patch_size, padding="same")


    def test_padding_wrong__type_type__error(self):

        # Arrange
        padding = None

        # Act and Assert
        with self.assertRaises(TypeError):
            Unfold(embedding_dim=1, patch_size=(1, 1, 1), padding=padding)

    
    def test_padding_wrong__value_value__error(self):
        
        # Arrange
        padding = "test"

        # Act and Assert
        with self.assertRaises(ValueError):
            Unfold(embedding_dim=1, patch_size=(1, 1, 1), padding=padding)


    def test_input__error_not__tensor_type__error(self):

        # Arrange
        input_tensor = [[[[[1, 2, 3]]]]]

        # Act and Assert
        with self.assertRaises(TypeError):
            Unfold(embedding_dim=1, patch_size=(1, 1, 1), padding="same")(input_tensor)


    def test_output_tensor_intended__shape(self):

        # Arrange
        input_tensor = tf.random.normal((1, 1, 1, 1, 3))

        # Act
        output = Unfold(embedding_dim=1, patch_size=(1, 1, 1), padding="same")(input_tensor)

        # Assert
        self.assertEqual(output.shape, (input_tensor.shape[0], input_tensor.shape[1] * 
                                        input_tensor.shape[2] * input_tensor.shape[3], 1))


    def test_get__config_layer_non__empty__dict(self):

        # Arrange and Act
        config = Unfold(embedding_dim=1, patch_size=(1, 1, 1), padding="same").get_config()

        # Assert
        self.assertGreater(len(config), 0)


    def test_from__config_layer__config__dict_input__config__dict(self):

        # Arrange
        config = Unfold(embedding_dim=1, patch_size=(1, 1, 1), padding="same").get_config()

        # Act
        cloned_layer = Unfold(embedding_dim=1, patch_size=(1, 1, 1), padding="same").from_config(config)

        # Assert
        self.assertEqual(config, cloned_layer.get_config())


if __name__ == "__main__":
    unittest.main()
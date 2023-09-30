import unittest
from MobileViViT.assets.layers.fold import Fold
import tensorflow as tf


class TestFold(unittest.TestCase):

    def test_embedding__dim_wrong__type_type__error(self):

        # Arrange
        embedding_dim = None

        # Act and Assert
        with self.assertRaises(TypeError):
            Fold(embedding_dim=embedding_dim, shape=(1, 1, 1, 1, 1))


    def test_embedding__dim_wrong__value_value__error(self):
        
        # Arrange
        embedding_dim = -1

        # Act and Assert
        with self.assertRaises(ValueError):
            Fold(embedding_dim=embedding_dim, shape=(1, 1, 1, 1, 1))


    def test_shape_wrong__type_type__error(self):

        # Arrange
        shape = 1

        # Act and Assert
        with self.assertRaises(TypeError):
            Fold(embedding_dim=1, shape=shape)


    def test_shape_not__len__five_value__error(self):

        # Arrange
        shape = (1, 1)

        # Act and Assert
        with self.assertRaises(ValueError):
            Fold(embedding_dim=1, shape=shape)


    def test_shape_wrong__value_value__error(self):

        # Arrange
        shape = (-1, -1, -1, -1, -1)

        # Act and Assert
        with self.assertRaises(ValueError):
            Fold(embedding_dim=1, shape=shape)


    def test_input__error_not__tensor_type__error(self):

        # Arrange
        input_tensor = [[[[[1, 2, 3]]]]]

        # Act and Assert
        with self.assertRaises(TypeError):
            Fold(embedding_dim=1, shape=(1, 1, 1, 1, 1))(input_tensor)


    def test_output_tensor_intended__shape(self):

        # Arrange
        input_tensor = tf.random.normal((1, 1, 1))

        # Act
        output = Fold(embedding_dim=1, shape=(1, 1, 1, 1, 1))(input_tensor)

        # Assert
        self.assertEqual(output.shape, (1, 1, 1, 1, 1))


    def test_get__config_layer_non__empty__dict(self):

        # Arrange and Act
        config = Fold(embedding_dim=1, shape=(1, 1, 1, 1, 1)).get_config()

        # Assert
        self.assertGreater(len(config), 0)


    def test_from__config_layer__config__dict_input__config__dict(self):

        # Arrange
        config = Fold(embedding_dim=1, shape=(1, 1, 1, 1, 1)).get_config()

        # Act
        clone_layer = Fold(embedding_dim=1, shape=(1, 1, 1, 1, 1)).from_config(config)

        # Assert
        self.assertEqual(config, clone_layer.get_config())


if __name__ == "__main__":
    unittest.main()
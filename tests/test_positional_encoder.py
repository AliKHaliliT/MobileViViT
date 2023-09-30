import unittest
from MobileViViT .assets.layers.positional_encoder import PositionalEncoder
import tensorflow as tf


class TestPositionalEncoder(unittest.TestCase):

    def test_embedding__dim_wrong__type_type__error(self):

        # Arrange
        embedding_dim = None

        # Act and Assert
        with self.assertRaises(TypeError):
            PositionalEncoder(embedding_dim=embedding_dim)


    def test_embedding__dim_wrong__value_value__error(self):
        
        # Arrange
        embedding_dim = -1

        # Act and Assert
        with self.assertRaises(ValueError):
            PositionalEncoder(embedding_dim=embedding_dim)


    def test_input__error_not__tensor_type__error(self):

        # Arrange
        input_tensor = [[[[[1, 2, 3]]]]]

        # Act and Assert
        with self.assertRaises(TypeError):
            PositionalEncoder(embedding_dim=1)(input_tensor)


    def test_output_tensor_intended__shape(self):

        # Arrange
        input_tensor = tf.random.normal((1, 1, 1))

        # Act
        output = PositionalEncoder(embedding_dim=1)(input_tensor)

        # Assert
        self.assertEqual(output.shape, input_tensor.shape)


    def test_get__config_layer_non__empty__dict(self):

        # Arrange and Act
        config = PositionalEncoder(embedding_dim=1).get_config()

        # Assert
        self.assertGreater(len(config), 0)


    def test_from__config_layer__config__dict_input__config__dict(self):

        # Arrange
        config = PositionalEncoder(embedding_dim=1).get_config()

        # Act
        cloned_layer = PositionalEncoder(embedding_dim=1).from_config(config)

        # Assert
        self.assertEqual(config, cloned_layer.get_config())


if __name__ == "__main__":
    unittest.main()
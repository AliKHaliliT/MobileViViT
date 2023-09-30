import unittest
from MobileViViT.assets.activations.mish import Mish
import tensorflow as tf


class TestMish(unittest.TestCase):

    def test_input__error_not__tensor_type__error(self):

        # Arrange
        input_tensor = [[[[[1, 2, 3]]]]]

        # Act and Assert
        with self.assertRaises(TypeError):
            Mish()(input_tensor)


    def test_output_tensor_intended__shape(self):

        # Arrange
        input_tensor = tf.random.normal((1, 1, 1, 1, 3))

        # Act
        output = Mish()(input_tensor)

        # Assert
        self.assertEqual(output.shape, input_tensor.shape)


    def test_get__config_layer_non__empty__dict(self):

        # Arrange and Act
        config = Mish().get_config()

        # Assert
        self.assertGreater(len(config), 0)


    def test_from__config_layer__config__dict_input__config__dict(self):

        # Arrange
        config = Mish().get_config()

        # Act
        clone_layer = Mish().from_config(config)

        # Assert
        self.assertEqual(config, clone_layer.get_config())


if __name__ == "__main__":
    unittest.main()
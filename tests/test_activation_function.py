import unittest
from MobileViViT.assets.utils.activation_function import activation_function
import tensorflow as tf


class TestActivationFunction(unittest.TestCase):

    def test_activation_wrong__type_type__error(self):

        # Arrange
        activation = None

        # Act and Assert
        with self.assertRaises(TypeError):
            activation_function(activation=activation)
    

    def test_activation_wrong__value_value__error(self):

        # Arrange
        activation = "test"

        # Act and Assert
        with self.assertRaises(ValueError):
            activation_function(activation=activation)


    def test_output_activation__name_layer(self):

        # Arrange
        activation = "relu"

        # Act
        output = activation_function(activation=activation)

        # Assert
        self.assertTrue(isinstance(output, tf.keras.layers.Layer))


if __name__ == "__main__":
    unittest.main()
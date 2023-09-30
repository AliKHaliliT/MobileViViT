import unittest
from MobileViViT.assets.activations.sine import Sine
import tensorflow as tf


class TestSine(unittest.TestCase):

    def test_omega__0_wrong__type_type__error(self):

        # Arrange
        omega_0 = None

        # Act and Assert
        with self.assertRaises(TypeError):
            Sine(omega_0=omega_0)
            

    def test_input__error_not__tensor_type__error(self):

        # Arrange
        input_tensor = [[[[[1, 2, 3]]]]]

        # Act and Assert
        with self.assertRaises(TypeError):
            Sine(omega_0=1.0)(input_tensor)


    def test_output_tensor_intended__shape(self):

        # Arrange
        input_tensor = tf.random.normal((1, 1, 1, 1, 3))

        # Act
        output = Sine(omega_0=1.0)(input_tensor)

        # Assert
        self.assertEqual(output.shape, input_tensor.shape)


    def test_get__config_layer_non__empty__dict(self):

        # Arrange and Act
        config = Sine(omega_0=1.0).get_config()

        # Assert
        self.assertGreater(len(config), 0)


    def test_from__config_layer__config__dict_input__config__dict(self):

        # Arrange
        config = Sine(omega_0=1.0).get_config()

        # Act
        cloned_layer = Sine(omega_0=1.0).from_config(config)

        # Assert
        self.assertEqual(config, cloned_layer.get_config())


if __name__ == "__main__":
    unittest.main()
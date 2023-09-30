import unittest
from MobileViViT.assets.layers.sine_layer import SineLayer
import tensorflow as tf


class TestSineLayer(unittest.TestCase):

    def test_units_wrong__type_type__error(self):

        # Arrange
        units = None

        # Act and Assert
        with self.assertRaises(TypeError):
            SineLayer(units=units, is_first=True, omega_0=30.0)


    def test_units_wrong__value_value__error(self):
        
        # Arrange
        units = -1

        # Act and Assert
        with self.assertRaises(ValueError):
            SineLayer(units=units, is_first=True, omega_0=30.0)


    def test_is__first_wrong__type_type__error(self):

        # Arrange
        is_first = "test"

        # Act and Assert
        with self.assertRaises(TypeError):
            SineLayer(units=1, is_first=is_first, omega_0=30.0)


    def test_omega__0_wrong__type_type__error(self):

        # Arrange
        omega_0 = None

        # Act and Assert
        with self.assertRaises(TypeError):
            SineLayer(units=1, is_first=True, omega_0=omega_0)


    def test_dropout_wrong__type_type__error(self):

        # Arrange
        dropout = "test"

        # Act and Assert
        with self.assertRaises(TypeError):
            SineLayer(units=1, is_first=True, omega_0=30.0, dropout=dropout)


    def test_dropout_wrong__value_value__error(self):

        # Arrange
        dropout = 1.2

        # Act and Assert
        with self.assertRaises(ValueError):
            SineLayer(units=1, is_first=True, omega_0=30.0, dropout=dropout)


    def test_input__error_not__tensor_type__error(self):

        # Arrange
        input_tensor = [[1, 2, 3]]

        # Act and Assert
        with self.assertRaises(TypeError):
            SineLayer(units=1, is_first=True, omega_0=30.0)(input_tensor)


    def test_output_tensor_intended__shape(self):

        # Arrange
        input_tensor = tf.random.normal((1, 1))

        # Act
        output = SineLayer(units=1, is_first=True, omega_0=30.0)(input_tensor)

        # Assert
        self.assertEqual(output.shape, input_tensor.shape)


    def test_get__config_layer_non__empty__dict(self):

        # Arrange and Act
        config = SineLayer(units=1, is_first=True, omega_0=30.0).get_config()

        # Assert
        self.assertGreater(len(config), 0)


    def test_from__config_layer__config__dict_input__config__dict(self):

        # Arrange
        config = SineLayer(units=1, is_first=True, omega_0=30.0).get_config()

        # Act
        cloned_layer = SineLayer(units=1, is_first=True, omega_0=30.0).from_config(config)

        # Assert
        self.assertEqual(config, cloned_layer.get_config())


if __name__ == "__main__":
    unittest.main()
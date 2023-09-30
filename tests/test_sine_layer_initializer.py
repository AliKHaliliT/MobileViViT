import unittest
from MobileViViT.assets.utils.sine_layer_initializer import SineLayerInitializer
import tensorflow as tf


class TestSineLayerInitializer(unittest.TestCase):

    def test_units_wrong__type_type__error(self):

        # Arrange
        units = None

        # Act and Assert
        with self.assertRaises(TypeError):
            SineLayerInitializer(units=units, is_first=True, omega_0=1.0)


    def test_units_wrong__value_value__error(self):
        
        # Arrange
        units = -1

        # Act and Assert
        with self.assertRaises(ValueError):
            SineLayerInitializer(units=units, is_first=True, omega_0=1.0)


    def test_is__first_wrong__type_type__error(self):

        # Arrange
        is_first = "test"

        # Act and Assert
        with self.assertRaises(TypeError):
            SineLayerInitializer(units=1, is_first=is_first, omega_0=1.0)

        
    def test_omega__0_wrong__type_type__error(self):

        # Arrange
        omega_0 = None

        # Act and Assert
        with self.assertRaises(TypeError):
            SineLayerInitializer(units=1, is_first=True, omega_0=omega_0)


    def test_output_shape_intended__shape(self):

        # Arrange
        shape = tf.TensorShape((1, 1))

        # Act
        output = SineLayerInitializer(units=1, is_first=True, omega_0=1.0)(shape=shape)

        # Assert
        self.assertEqual(output.shape, shape)


    def test_get__config_layer_non__empty__dict(self):

        # Arrange and Act
        config = SineLayerInitializer(units=1, is_first=True, omega_0=1.0).get_config()

        # Assert
        self.assertGreater(len(config), 0)


    def test_from__config_layer__config__dict_input__config__dict(self):

        # Arrange
        config = SineLayerInitializer(units=1, is_first=True, omega_0=1.0).get_config()

        # Act
        cloned_layer = SineLayerInitializer(units=1, is_first=True, omega_0=1.0).from_config(config)

        # Assert
        self.assertEqual(config, cloned_layer.get_config())


if __name__ == "__main__":
    unittest.main()
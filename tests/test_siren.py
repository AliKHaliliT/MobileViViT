import unittest
from MobileViViT.assets.blocks.siren import SIREN
import tensorflow as tf


class TestSIREN(unittest.TestCase):

    def test_units__list_wrong__type_type__error(self):

        # Arrange
        units_list = None

        # Act and Assert
        with self.assertRaises(TypeError):
            SIREN(units_list=units_list)


    def test_units__list_empty__units__list_value__error(self):

        # Arrange
        units_list = []

        # Act and Assert
        with self.assertRaises(ValueError):
            SIREN(units_list=units_list)


    def test_units__list_wrong__type__instances_type__error(self):

        # Arrange
        units_list = [1, 1, None]

        # Act and Assert
        with self.assertRaises(TypeError):
            SIREN(units_list=units_list)


    def test_units__list_wrong__value_value__error(self):

        # Arrange
        units_list = [-1, -1, -1]

        # Act and Assert
        with self.assertRaises(ValueError):
            SIREN(units_list=units_list)


    def test_omega__0_wrong__type_type__error(self):

        # Arrange
        omega_0 = None

        # Act and Assert
        with self.assertRaises(TypeError):
            SIREN(units_list=[1, 1], omega_0=omega_0)


    def test_dropout_wrong__type_type__error(self):

        # Arrange
        dropout = "test"

        # Act and Assert
        with self.assertRaises(TypeError):
            SIREN(units_list=[1, 1], dropout=dropout)


    def test_dropout_wrong__value_value__error(self):

        # Arrange
        dropout = 1.2

        # Act and Assert
        with self.assertRaises(ValueError):
            SIREN(units_list=[1, 1], dropout=dropout)


    def test_input__error_not__tensor_type__error(self):

        # Arrange
        input_tensor = [[1, 2, 3]]

        # Act and Assert
        with self.assertRaises(TypeError):
            SIREN(units_list=[1, 1])(input_tensor)


    def test_output_tensor_intended__shape(self):

        # Arrange
        input_tensor = tf.random.normal((1, 1))

        # Act
        output = SIREN(units_list=[1, 1])(input_tensor)

        # Assert
        self.assertEqual(output.shape, input_tensor.shape)


    def test_get__config_layer_non__empty__dict(self):

        # Arrange and Act
        config = SIREN(units_list=[1, 1]).get_config()

        # Assert
        self.assertGreater(len(config), 0)


    def test_from__config_layer__config__dict_input__config__dict(self):

        # Arrange
        config = SIREN(units_list=[1, 1]).get_config()

        # Act
        cloned_layer = SIREN(units_list=[1, 1]).from_config(config)

        # Assert
        self.assertEqual(config, cloned_layer.get_config())


if __name__ == "__main__":
    unittest.main()
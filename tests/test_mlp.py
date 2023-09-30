import unittest
from MobileViViT.assets.blocks.mlp import MLP
import tensorflow as tf


class TestMLP(unittest.TestCase):

    def test_units__list_wrong__type_type__error(self):

        # Arrange
        units_list = None

        # Act and Assert
        with self.assertRaises(TypeError):
            MLP(units_list=units_list)


    def test_units__list_empty__units__list_value__error(self):

        # Arrange
        units_list = []

        # Act and Assert
        with self.assertRaises(ValueError):
            MLP(units_list=units_list)


    def test_units__list_wrong__type__instances_type__error(self):

        # Arrange
        units_list = [1, 1, None]

        # Act and Assert
        with self.assertRaises(TypeError):
            MLP(units_list=units_list)


    def test_units__list_wrong__value_value__error(self):

        # Arrange
        units_list = [-1, -1, -1]

        # Act and Assert
        with self.assertRaises(ValueError):
            MLP(units_list=units_list)


    def test_activation_wrong__type_type__error(self):

        # Arrange
        activation = None

        # Act and Assert
        with self.assertRaises(TypeError):
            MLP(units_list=[1, 1], activation=activation)

        
    def test_activation_wrong__value_value__error(self):

        # Arrange
        activation = "test"

        # Act and Assert
        with self.assertRaises(ValueError):
            MLP(units_list=[1, 1], activation=activation)


    def test_dropout_wrong__type_type__error(self):

        # Arrange
        dropout = "test"

        # Act and Assert
        with self.assertRaises(TypeError):
            MLP(units_list=[1, 1], dropout=dropout)


    def test_dropout_wrong__value_value__error(self):

        # Arrange
        dropout = 1.2

        # Act and Assert
        with self.assertRaises(ValueError):
            MLP(units_list=[1, 1], dropout=dropout)


    def test_input__error_not__tensor_type__error(self):

        # Arrange
        input_tensor = [[1, 2, 3]]

        # Act and Assert
        with self.assertRaises(TypeError):
            MLP(units_list=[1, 1])(input_tensor)


    def test_output_tensor_intended__shape(self):

        # Arrange
        input_tensor = tf.random.normal((1, 1))

        # Act
        output = MLP(units_list=[1, 1])(input_tensor)

        # Assert
        self.assertEqual(output.shape, input_tensor.shape)


    def test_get__config_layer_non__empty__dict(self):

        # Arrange and Act
        config = MLP(units_list=[1, 1]).get_config()

        # Assert
        self.assertGreater(len(config), 0)


    def test_from__config_layer__config__dict_input__config__dict(self):

        # Arrange
        config = MLP(units_list=[1, 1]).get_config()

        # Act
        cloned_layer = MLP(units_list=[1, 1]).from_config(config)

        # Assert
        self.assertEqual(config, cloned_layer.get_config())


if __name__ == "__main__":
    unittest.main()
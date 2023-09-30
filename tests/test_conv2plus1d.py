import unittest
from MobileViViT.assets.layers.conv2plus1d import Conv2Plus1D
import tensorflow as tf


class TestConv2Plus1D(unittest.TestCase):

    def test_filters_wrong__type_type__error(self):

        # Arrange
        filters = None

        # Act and Assert
        with self.assertRaises(TypeError):
            Conv2Plus1D(filters=filters, kernel_size=(1, 1, 1), 
                        strides=(1, 1, 1), padding="same")


    def test_filters_wrong__value_value__error(self):
        
        # Arrange
        filters = -1

        # Act and Assert
        with self.assertRaises(ValueError):
            Conv2Plus1D(filters=filters, kernel_size=(1, 1, 1), 
                        strides=(1, 1, 1), padding="same")


    def test_kernel__size_wrong__type_type__error(self):

        # Arrange
        kernel_size = None

        # Act and Assert
        with self.assertRaises(TypeError):
            Conv2Plus1D(filters=1, kernel_size=kernel_size, 
                        strides=(1, 1, 1), padding="same")


    def test_kernel__size_not__len__three_value__error(self):

        # Arrange
        kernel_size = (1, 1)

        # Act and Assert
        with self.assertRaises(ValueError):
            Conv2Plus1D(filters=1, kernel_size=kernel_size, 
                        strides=(1, 1, 1), padding="same")


    def test_kernel__size_wrong__type__instances_type__error(self):

        # Arrange
        kernel_size = (1, 1, None)

        # Act and Assert
        with self.assertRaises(TypeError):
            Conv2Plus1D(filters=1, kernel_size=kernel_size, 
                        strides=(1, 1, 1), padding="same")


    def test_kernel__size_wrong__value_value__error(self):

        # Arrange
        kernel_size = (-1, -1, -1)

        # Act and Assert
        with self.assertRaises(ValueError):
            Conv2Plus1D(filters=1, kernel_size=kernel_size, 
                        strides=(1, 1, 1), padding="same")

    
    def test_strides_wrong__type_type__error(self):

        # Arrange
        strides = None

        # Act and Assert
        with self.assertRaises(TypeError):
            Conv2Plus1D(filters=1, kernel_size=(1, 1, 1), 
                        strides=strides, padding="same")

    
    def test_strides_not__len__three_value__error(self):

        # Arrange
        strides = (1, 1)

        # Act and Assert
        with self.assertRaises(ValueError):
            Conv2Plus1D(filters=1, kernel_size=(1, 1), 
                        strides=strides, padding="same")


    def test_strides_wrong__type__instances_type__error(self):

        # Arrange
        strides = (1, 1, None)

        # Act and Assert
        with self.assertRaises(TypeError):
            Conv2Plus1D(filters=1, kernel_size=(1, 1, 1), 
                        strides=strides, padding="same")

    
    def test_strides_wrong__value_value__error(self):
        
        # Arrange
        strides = (-1, -1, -1)

        # Act and Assert
        with self.assertRaises(ValueError):
            Conv2Plus1D(filters=1, kernel_size=(0, 0, 0), 
                        strides=strides, padding="same")


    def test_padding_wrong__type_type__error(self):

        # Arrange
        padding = None

        # Act and Assert
        with self.assertRaises(TypeError):
            Conv2Plus1D(filters=1, kernel_size=(1, 1, 1), 
                        strides=(1, 1, 1), padding=padding)

    
    def test_padding_wrong__value_value__error(self):
        
        # Arrange
        padding = "test"

        # Act and Assert
        with self.assertRaises(ValueError):
            Conv2Plus1D(filters=1, kernel_size=(1, 1, 1), 
                        strides=(1, 1, 1), padding=padding)


    def test_input__error_not__tensor_type__error(self):

        # Arrange
        input_tensor = [[[[[1, 2, 3]]]]]

        # Act and Assert
        with self.assertRaises(TypeError):
            Conv2Plus1D(filters=1, kernel_size=(1, 1, 1), 
                        strides=(1, 1, 1), padding="same")(input_tensor)


    def test_output_tensor_intended__shape(self):

        # Arrange
        input_tensor = tf.random.normal((1, 1, 1, 1, 3))

        # Act
        output = Conv2Plus1D(filters=1, kernel_size=(1, 1, 1), 
                             strides=(1, 1, 1), padding="same")(input_tensor)

        # Assert
        self.assertEqual(output.shape, (1, 1, 1, 1, 1))


    def test_get__config_layer_non__empty__dict(self):

        # Arrange and Act
        config = Conv2Plus1D(filters=1, kernel_size=(1, 1, 1), 
                             strides=(1, 1, 1), padding="same").get_config()

        # Assert
        self.assertGreater(len(config), 0)


    def test_from__config_layer__config__dict_input__config__dict(self):

        # Arrange
        config = Conv2Plus1D(filters=1, kernel_size=(1, 1, 1), 
                             strides=(1, 1, 1), padding="same").get_config()

        # Act
        cloned_layer = Conv2Plus1D(filters=1, kernel_size=(1, 1, 1), 
                                   strides=(1, 1, 1), padding="same").from_config(config)

        # Assert
        self.assertEqual(config, cloned_layer.get_config())


if __name__ == "__main__":
    unittest.main()
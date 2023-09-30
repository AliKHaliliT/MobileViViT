import unittest
from MobileViViT.assets.layers.conv_layer import ConvLayer
import tensorflow as tf


class TestConvLayer(unittest.TestCase):

    def test_filters_wrong__type_type__error(self):

        # Arrange
        filters = None

        # Act and Assert
        with self.assertRaises(TypeError):
            ConvLayer(filters=filters, kernel_size=(1, 1, 1), strides=(1, 1, 1), 
                      padding="same", batch_norm=False, activation="relu")


    def test_filters_wrong__value_value__error(self):
        
        # Arrange
        filters = -1

        # Act and Assert
        with self.assertRaises(ValueError):
            ConvLayer(filters=filters, kernel_size=(1, 1, 1), strides=(1, 1, 1), 
                      padding="same", batch_norm=False, activation="relu")


    def test_kernel__size_wrong__type_type__error(self):

        # Arrange
        kernel_size = None

        # Act and Assert
        with self.assertRaises(TypeError):
            ConvLayer(filters=1, kernel_size=kernel_size, strides=(1, 1, 1), 
                      padding="same", batch_norm=False, activation="relu")


    def test_kernel__size_not__len__three_value__error(self):

        # Arrange
        kernel_size = (1, 1)

        # Act and Assert
        with self.assertRaises(ValueError):
            ConvLayer(filters=1, kernel_size=kernel_size, strides=(1, 1, 1), 
                      padding="same", batch_norm=False, activation="relu")


    def test_kernel__size_wrong__type__instances_type__error(self):

        # Arrange
        kernel_size = (1, 1, None)

        # Act and Assert
        with self.assertRaises(TypeError):
            ConvLayer(filters=1, kernel_size=kernel_size, strides=(1, 1, 1), 
                      padding="same", batch_norm=False, activation="relu")


    def test_kernel__size_wrong__value_value__error(self):

        # Arrange
        kernel_size = (-1, -1, -1)

        # Act and Assert
        with self.assertRaises(ValueError):
            ConvLayer(filters=1, kernel_size=kernel_size, strides=(1, 1, 1), 
                      padding="same", batch_norm=False, activation="relu")

    
    def test_strides_wrong__type_type__error(self):

        # Arrange
        strides = None

        # Act and Assert
        with self.assertRaises(TypeError):
            ConvLayer(filters=1, kernel_size=(1, 1, 1), strides=strides, 
                      padding="same", batch_norm=False, activation="relu")

    
    def test_strides_not__len__three_value__error(self):

        # Arrange
        strides = (1, 1)

        # Act and Assert
        with self.assertRaises(ValueError):
            ConvLayer(filters=1, kernel_size=(1, 1), strides=strides, 
                      padding="same", batch_norm=False, activation="relu")


    def test_strides_wrong__type__instances_type__error(self):

        # Arrange
        strides = (1, 1, None)

        # Act and Assert
        with self.assertRaises(TypeError):
            ConvLayer(filters=1, kernel_size=(1, 1, 1), strides=strides, 
                      padding="same", batch_norm=False, activation="relu")

    
    def test_strides_wrong__value_value__error(self):
        
        # Arrange
        strides = (-1, -1, -1)

        # Act and Assert
        with self.assertRaises(ValueError):
            ConvLayer(filters=1, kernel_size=(1, 1, 1), strides=strides, 
                      padding="same", batch_norm=False, activation="relu")


    def test_padding_wrong__type_type__error(self):

        # Arrange
        padding = None

        # Act and Assert
        with self.assertRaises(TypeError):
            ConvLayer(filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1), 
                      padding=padding, batch_norm=False, activation="relu")

    
    def test_padding_wrong__value_value__error(self):
        
        # Arrange
        padding = "test"

        # Act and Assert
        with self.assertRaises(ValueError):
            ConvLayer(filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1), 
                      padding=padding, batch_norm=False, activation="relu")


    def test_batch__norm_wrong__type_type__error(self):
        
        # Arrange
        batch_norm = None

        # Act and Assert
        with self.assertRaises(TypeError):
            ConvLayer(filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1), 
                      padding="same", batch_norm=batch_norm, activation="relu")


    def test_activation_wrong__type_type__error(self):

        # Arrange
        activation = 1

        # Act and Assert
        with self.assertRaises(TypeError):
            ConvLayer(filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1), 
                      padding="same", batch_norm=False, activation=activation)

        
    def test_activation_wrong__value_value__error(self):

        # Arrange
        activation = "test"

        # Act and Assert
        with self.assertRaises(ValueError):
            ConvLayer(filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1), 
                      padding="same", batch_norm=False, activation=activation)


    def test_input__error_not__tensor_type__error(self):

        # Arrange
        input_tensor = [[[[[1, 2, 3]]]]]

        # Act and Assert
        with self.assertRaises(TypeError):
            ConvLayer(filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1), 
                      padding="same", batch_norm=False, activation="relu")(input_tensor)


    def test_output_tensor_intended__shape(self):

        # Arrange
        input_tensor = tf.random.normal((1, 1, 1, 1, 3))

        # Act
        output = ConvLayer(filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1), 
                           padding="same", batch_norm=False, activation="relu")(input_tensor)

        # Assert
        self.assertEqual(output.shape, (1, 1, 1, 1, 1))


    def test_get__config_layer_non__empty__dict(self):

        # Arrange and Act
        config = ConvLayer(filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1), 
                           padding="same", batch_norm=False, activation="relu").get_config()

        # Assert
        self.assertGreater(len(config), 0)


    def test_from__config_layer__config__dict_input__config__dict(self):

        # Arrange
        config = ConvLayer(filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1), 
                           padding="same", batch_norm=False, activation="relu").get_config()

        # Act
        cloned_layer = ConvLayer(filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1), 
                                 padding="same", batch_norm=False, activation="relu").from_config(config)

        # Assert
        self.assertEqual(config, cloned_layer.get_config())


if __name__ == "__main__":
    unittest.main()
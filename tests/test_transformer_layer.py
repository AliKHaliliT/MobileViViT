import unittest
from MobileViViT.assets.layers.transformer_layer import TransformerLayer
import tensorflow as tf


class TestTransformerLayer(unittest.TestCase):

    def test_projection__dim_wrong__type_type__error(self):

        # Arrange
        projection_dim = None

        # Act and Assert
        with self.assertRaises(TypeError):
            TransformerLayer(projection_dim=projection_dim, num_transformer_heads=1)


    def test_projection__dim_wrong__value_value__error(self):
        
        # Arrange
        projection_dim = -1

        # Act and Assert
        with self.assertRaises(ValueError):
            TransformerLayer(projection_dim=projection_dim, num_transformer_heads=1)

    
    def test_num__transformer__heads_wrong__type_type__error(self):
        
        # Arrange
        num_transformer_heads = None

        # Act and Assert
        with self.assertRaises(TypeError):
            TransformerLayer(projection_dim=1, num_transformer_heads=num_transformer_heads)

    
    def test_num__transformer__heads_wrong__value_value__error(self):

        # Arrange
        num_transformer_heads = -1

        # Act and Assert
        with self.assertRaises(ValueError):
            TransformerLayer(projection_dim=1, num_transformer_heads=num_transformer_heads)


    def test_dropout__mha_wrong__type_type__error(self):

        # Arrange
        dropout_mha = "test"

        # Act and Assert
        with self.assertRaises(TypeError):
            TransformerLayer(projection_dim=1, num_transformer_heads=1, dropout_mha=dropout_mha)


    def test_dropout__mha_wrong__value_value__error(self):

        # Arrange
        dropout_mha = 1.2

        # Act and Assert
        with self.assertRaises(ValueError):
            TransformerLayer(projection_dim=1, num_transformer_heads=1, dropout_mha=dropout_mha)


    def test_aproximator__type_wrong__type_type__error(self):
        
        # Arrange
        aproximator_type = None

        # Act and Assert
        with self.assertRaises(TypeError):
            TransformerLayer(projection_dim=1, num_transformer_heads=1, aproximator_type=aproximator_type)


    def test_aproximator__type_wrong__value_value__error(self):

        # Arrange
        aproximator_type = "test"

        # Act and Assert
        with self.assertRaises(ValueError):
            TransformerLayer(projection_dim=1, num_transformer_heads=1, aproximator_type=aproximator_type)


    def test_activation_wrong__type_type__error(self):

        # Arrange
        activation = None

        # Act and Assert
        with self.assertRaises(TypeError):
            TransformerLayer(projection_dim=1, num_transformer_heads=1, activation=activation)

        
    def test_activation_wrong__value_value__error(self):

        # Arrange
        activation = "test"

        # Act and Assert
        with self.assertRaises(ValueError):
            TransformerLayer(projection_dim=1, num_transformer_heads=1, activation=activation)


    def test_dropout__mlp_wrong__type_type__error(self):

        # Arrange
        dropout_mlp = "test"

        # Act and Assert
        with self.assertRaises(TypeError):
            TransformerLayer(projection_dim=1, num_transformer_heads=1, dropout_mlp=dropout_mlp)


    def test_dropout__mlp_wrong__value_value__error(self):

        # Arrange
        dropout_mlp = 1.2

        # Act and Assert
        with self.assertRaises(ValueError):
            TransformerLayer(projection_dim=1, num_transformer_heads=1, dropout_mlp=dropout_mlp)

    def test_input__error_not__tensor_type__error(self):

        # Arrange
        input_tensor = [[[[[1, 2, 3]]]]]

        # Act and Assert
        with self.assertRaises(TypeError):
            TransformerLayer(projection_dim=1, num_transformer_heads=1)(input_tensor)


    def test_output_tensor_intended__shape(self):

        # Arrange
        input_tensor = tf.random.normal((1, 1, 1))

        # Act
        output = TransformerLayer(projection_dim=1, num_transformer_heads=1)(input_tensor)

        # Assert
        self.assertEqual(output.shape, input_tensor.shape)


    def test_get__config_layer_non__empty__dict(self):

        # Arrange and Act
        config = TransformerLayer(projection_dim=1, num_transformer_heads=1).get_config()

        # Assert
        self.assertGreater(len(config), 0)


    def test_from__config_layer__config__dict_input__config__dict(self):

        # Arrange
        config = TransformerLayer(projection_dim=1, num_transformer_heads=1).get_config()

        # Act
        cloned_layer = TransformerLayer(projection_dim=1, num_transformer_heads=1).from_config(config)

        # Assert
        self.assertEqual(config, cloned_layer.get_config())


if __name__ == "__main__":
    unittest.main()
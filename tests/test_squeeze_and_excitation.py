import unittest
from MobileViViT.assets.utils.squeeze_and_excitation import squeeze_and_excitation
import tensorflow as tf


class TestActivationFunction(unittest.TestCase):

    def test_sae__type_wrong__type_type__error(self):

        # Arrange
        sae_type = None

        # Act and Assert
        with self.assertRaises(TypeError):
            squeeze_and_excitation(sae_type=sae_type)

    
    def test_sae__type_wrong__value_value__error(self):

        # Arrange
        sae_type = "test"

        # Act and Assert
        with self.assertRaises(ValueError):
            squeeze_and_excitation(sae_type=sae_type)


    def test_reduction__ratio_wrong__type_type__error(self):

        # Arrange
        reduction_ratio = None

        # Act and Assert
        with self.assertRaises(TypeError):
            squeeze_and_excitation(sae_type="vanilla", reduction_ratio=reduction_ratio)


    def test_reduction__ratio_wrong__value_value__error(self):
        
        # Arrange
        reduction_ratio = -1

        # Act and Assert
        with self.assertRaises(ValueError):
            squeeze_and_excitation(sae_type="vanilla", reduction_ratio=reduction_ratio)


    def test_activation_wrong__type_type__error(self):

        # Arrange
        activation = None

        # Act and Assert
        with self.assertRaises(TypeError):
            squeeze_and_excitation(sae_type="vanilla", activation=activation)

        
    def test_activation_wrong__value_value__error(self):

        # Arrange
        activation = "test"

        # Act and Assert
        with self.assertRaises(ValueError):
            squeeze_and_excitation(sae_type="vanilla", activation=activation)


    def test_omega__0_wrong__type_type__error(self):

        # Arrange
        omega_0 = None

        # Act and Assert
        with self.assertRaises(TypeError):
            squeeze_and_excitation(sae_type="vanilla", omega_0=omega_0)


    def test_output_squeeze__and__excitation__name_layer(self):

        # Arrange
        sae_type = "vanilla"

        # Act
        output = squeeze_and_excitation(sae_type=sae_type)

        # Assert
        self.assertTrue(isinstance(output, tf.keras.layers.Layer))


if __name__ == "__main__":
    unittest.main()
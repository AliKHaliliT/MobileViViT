import unittest
from MobileViViT import MobileViViTS
import tensorflow as tf


class TestMobileViViTS(unittest.TestCase):

    def test_activation_wrong__type_type__error(self):

        # Arrange
        num_output_units = None

        # Act and Assert
        with self.assertRaises(TypeError):
            MobileViViTS(num_output_units=num_output_units)
    

    def test_activation_wrong__value_value__error(self):

        # Arrange
        num_output_units = -1

        # Act and Assert
        with self.assertRaises(ValueError):
            MobileViViTS(num_output_units=num_output_units)


    def test_output_initialization_model(self):

        # Arrange and Act
        output = MobileViViTS(num_output_units=1)

        # Assert
        self.assertTrue(isinstance(output, tf.keras.Model))


    def test_compile_initialization_model(self):

        # Arrange
        output = MobileViViTS(num_output_units=1)

        # Act
        output.compile(optimizer="adam", loss="mse")

        # Assert
        self.assertTrue(output.optimizer is not None)


    # Too burdensome to test
    # def test_fit_initialization_model(self):

    #     # Arrange
    #     X = tf.random.uniform((1, 50, 256, 256, 3))
    #     y = tf.random.uniform((1, 1))
    #     output = MobileViViTS(num_output_units=1)

    #     # Act
    #     output.compile(optimizer="adam", loss="mse")
    #     history = output.fit(X, y, epochs=1)

    #     # Assert
    #     self.assertIsInstance(history, tf.keras.callbacks.History)


    # def test_evaluate_initialization_model(self):

    #     # Arrange
    #     X = tf.random.uniform((1, 50, 256, 256, 3))
    #     y = tf.random.uniform((1, 1))
    #     output = MobileViViTS(num_output_units=1)

    #     # Act
    #     output.compile(optimizer="adam", loss="mse")
    #     output.fit(X, y, epochs=1)
    #     loss = output.evaluate(X, y)

    #     # Assert
    #     self.assertTrue(loss is not None)


if __name__ == "__main__":
    unittest.main()
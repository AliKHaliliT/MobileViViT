import unittest
from MobileViViT.assets.utils.progress_bar import progress_bar


class TestProgressBar(unittest.TestCase):

    def test_current_wrong__type_type__error(self):

        # Arrange
        current = None

        # Act and Assert
        with self.assertRaises(TypeError):
            progress_bar(current=current, total=1)
    

    def test_current_wrong__value_value__error(self):

        # Arrange
        current = -1

        # Act and Assert
        with self.assertRaises(ValueError):
            progress_bar(current=current, total=1)


    def test_total_wrong__type_type__error(self):

        # Arrange
        total = None

        # Act and Assert
        with self.assertRaises(TypeError):
            progress_bar(current=1, total=total)
    

    def test_total_wrong__value_value__error(self):

        # Arrange
        total = -1

        # Act and Assert
        with self.assertRaises(ValueError):
            progress_bar(current=1, total=total)


    def test_bar__length_wrong__type_type__error(self):

        # Arrange
        bar_length = None

        # Act and Assert
        with self.assertRaises(TypeError):
            progress_bar(current=1, total=1, bar_length=bar_length)
    

    def test_bar__length_wrong__value_value__error(self):

        # Arrange
        bar_length = -1

        # Act and Assert
        with self.assertRaises(ValueError):
            progress_bar(current=1, total=1, bar_length=bar_length)


    def test_description_wrong__type_type__error(self):

        # Arrange
        description = None

        # Act and Assert
        with self.assertRaises(TypeError):
            progress_bar(current=1, total=1, description=description)


    def test_output_progress__parameters_str(self):

        # Arrange and Act
        output = progress_bar(current=1, total=1)

        # Assert
        self.assertTrue(isinstance(output, str))


if __name__ == "__main__":
    unittest.main()
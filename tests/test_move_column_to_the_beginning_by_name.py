import unittest
from MobileViViT.assets.utils.move_column_to_the_beginning import MoveColumnToTheBeginning as movetobeginning
import pandas as pd


class TestMoveColumnToTheBeginningByName(unittest.TestCase):

    def test_dataframe_wrong__type_type__error(self):

        # Arrange
        dataframe = None

        # Act and Assert
        with self.assertRaises(TypeError):
            movetobeginning.move_column_to_the_beginning_by_name(dataframe=dataframe, column_name="test")


    def test_column__name__wrong_type__type_error(self):

        # Arrange
        dataframe = pd.DataFrame({'0': [1, 2, 3]})
        column_name = None

        # Act and Assert
        with self.assertRaises(TypeError):
            movetobeginning.move_column_to_the_beginning_by_name(dataframe=dataframe, column_name=column_name)            


    def test_column__name_wrong__value_value__error(self):

        # Arrange
        dataframe = pd.DataFrame({'0': [1, 2, 3]})
        column_name = "test"

        # Act and Assert
        with self.assertRaises(ValueError):
            movetobeginning.move_column_to_the_beginning_by_name(dataframe=dataframe, column_name=column_name)


    def test_output_dataframe_dataframe(self):

        # Arrange
        dataframe = pd.DataFrame({'0': [1, 2, 3], '1': [4, 5, 6], '2': [7, 8, 9]})
        column_name = '0'
        
        # Act
        output = movetobeginning.move_column_to_the_beginning_by_name(dataframe=dataframe, column_name=column_name)

        # Assert
        self.assertTrue(isinstance(output, pd.DataFrame))


if __name__ == "__main__":
    unittest.main()
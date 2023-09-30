import unittest
from MobileViViT.assets.utils.move_column_to_the_beginning import MoveColumnToTheBeginning as movetobeginning
import pandas as pd


class TestMoveColumnToTheBeginningByIndex(unittest.TestCase):

    def test_dataframe_wrong__type_type__error(self):

        # Arrange
        dataframe = None

        # Act and Assert
        with self.assertRaises(TypeError):
            movetobeginning.move_column_to_the_beginning_by_index(dataframe=dataframe, column_index=1)


    def test_column__index__wrong_type__type_error(self):

        # Arrange
        dataframe = pd.DataFrame({'0': [1, 2, 3]})
        column_index = None

        # Act and Assert
        with self.assertRaises(TypeError):
            movetobeginning.move_column_to_the_beginning_by_index(dataframe=dataframe, column_index=column_index)            


    def test_column__index_wrong__value_value__error(self):

        # Arrange
        dataframe = pd.DataFrame({'0': [1, 2, 3]})
        column_index = -1

        # Act and Assert
        with self.assertRaises(ValueError):
            movetobeginning.move_column_to_the_beginning_by_index(dataframe=dataframe, column_index=column_index)


    def test_output_dataframe_dataframe(self):

        # Arrange
        dataframe = pd.DataFrame({'0': [1, 2, 3], '1': [4, 5, 6], '2': [7, 8, 9]})
        column_index = 0
        
        # Act
        output = movetobeginning.move_column_to_the_beginning_by_index(dataframe=dataframe, column_index=column_index)

        # Assert
        self.assertTrue(isinstance(output, pd.DataFrame))


if __name__ == "__main__":
    unittest.main()
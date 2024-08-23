import pandas as pd
from .move_column_to_the_beginning_assets._by_name import _by_name
from .move_column_to_the_beginning_assets._by_index import _by_index


class MoveColumnToTheBeginning:

    """

    A class containing functions to move a column to the beginning of a Pandas DataFrame.

    """

    def __init__(self) -> None:

        """

        Constructor of the MoveColumnToTheBeginning class.

        
        Parameters
        ----------
        None.

        
        Returns
        -------
        None.

        """

        pass

        
    @staticmethod
    def move_column_to_the_beginning_by_name(dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:

        """
        
        Moves a column to the beginning of a Pandas DataFrame
        based on its name.
        

        Parameters
        ----------
        dataframe : pandas.DataFrame
            DataFrame containing the column to be moved.

        column_name : str
            Name of the column to be moved.


        Returns
        -------
        dataframe : pandas.DataFrame
            DataFrame with the column moved to the beginning.
        
        """

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a DataFrame")
        if not isinstance(column_name, str):
            raise TypeError("column_name must be a string")
        if column_name not in dataframe.columns:
            raise ValueError("column_name must be a column of the DataFrame")
        

        return _by_name(dataframe, column_name)


    @staticmethod
    def move_column_to_the_beginning_by_index(dataframe: pd.DataFrame, column_index: int) -> pd.DataFrame:

        """

        Moves a column to the beginning of a Pandas DataFrame 
        based on its index (0-based index).
        

        Parameters
        ----------
        dataframe : pandas.DataFrame
            DataFrame containing the column to be moved.
        
        column_index : int
            Index of the column to be moved (0-based index).
        
            
        Returns
        -------
        dataframe : pandas.DataFrame
            DataFrame with the column moved to the beginning.

        """
        
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a DataFrame")
        if not isinstance(column_index, int):
            raise TypeError("column_index must be an integer")
        if column_index < 0 or column_index >= len(dataframe.columns):
            raise ValueError("column_index is out of range for DataFrame")


        return _by_index(dataframe, column_index)
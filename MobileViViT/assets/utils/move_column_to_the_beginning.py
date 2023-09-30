import pandas as pd


class MoveColumnToTheBeginning:

    """

    A class containing functions to move a column to the beginning of a Pandas DataFrame.

    """
        
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
        

        if column_name == dataframe.columns[0]:

            print("The column is already in the beginning, returning the original DataFrame.")


            return dataframe
        
        else:
                
            # Get the column
            column = dataframe[column_name]

            # Remove the column from its current position
            dataframe = dataframe.drop(columns=column_name)

            # Add the column to the beginning
            dataframe.insert(0, column_name, column)


            return dataframe


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


        if column_index == 0:

            print("The column is already in the beginning, returning the original DataFrame.")


            return dataframe
        
        else:

            # Get the column
            column = dataframe.iloc[:, column_index]

            # Remove the column from its current position
            dataframe = dataframe.drop(dataframe.columns[column_index], axis=1)

            # Add the column to the beginning
            dataframe.insert(0, dataframe.columns[column_index], column)


            return dataframe
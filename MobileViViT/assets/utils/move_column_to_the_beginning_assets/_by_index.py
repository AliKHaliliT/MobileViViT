import pandas as pd    


def _by_index(dataframe: pd.DataFrame, column_index: int) -> pd.DataFrame:

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
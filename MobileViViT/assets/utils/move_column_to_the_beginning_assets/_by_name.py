import pandas as pd


def _by_name(dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:

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
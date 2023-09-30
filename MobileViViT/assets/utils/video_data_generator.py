import tensorflow as tf
import pandas as pd
from typing import Optional, Union
import numpy as np
from .move_column_to_the_beginning import MoveColumnToTheBeginning as movetobeginning
from .video_file_to_numpy_array import video_file_to_numpy_array as videotoarray


class VideoDataGenerator(tf.keras.utils.Sequence):

    """
    
    A custom data generator for loading video frames and labels from a Pandas DataFrame.
    
    """

    def __init__(self, dataframe: pd.DataFrame, batch_size: int, 
                 shuffle: bool = True, 
                 path_col: Optional[Union[str, int]] = None,
                 dtype: np.dtype = np.float32) -> None:

        """
        
        Constructor of the VideoDataGenerator class.


        Parameters
        ----------
        dataframe : pandas.DataFrame
            DataFrame containing video file paths and labels. 
            The video file paths must be in the first column.
            If the video file paths are not in the first column, 
            you must specify the column index or name using the path_col parameter.

        batch_size : int
            Batch size.
        
        shuffle : bool, optional
            Whether to shuffle the indices after each epoch. The default is True.

        path_col : str or int, optional
            Name or index of the column containing the video file paths. 
            The default is None.

        dtype: numpy.dtype, optional
            Data type of the numpy array. The default value is np.float32.

        
        Returns
        -------
        None.

        """

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")
        if not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a boolean")
        if path_col is not None:
            if not isinstance(path_col, str) and not isinstance(path_col, int):
                raise TypeError("path_col must be a string or an integer")
        if dtype is None:
            raise TypeError("dtype must be a numpy.dtype")
        elif not np.issubdtype(dtype, np.generic):
            raise TypeError("dtype must be a numpy.dtype")    
        

        self.dataframe = dataframe
        self.batch_size = batch_size
        self.indexes = np.arange(len(dataframe))
        self.shuffle = shuffle
        self.path_col = path_col
        self.dtype = dtype

        if path_col is not None:
            if isinstance(path_col, str):
                self.dataframe = movetobeginning.move_column_to_the_beginning_by_name(dataframe, path_col)
            elif isinstance(path_col, int):
                self.dataframe = movetobeginning.move_column_to_the_beginning_by_index(dataframe, path_col)


    def __len__(self) -> int:

        """
        
        Method to calculate the number of batches per epoch.


        Parameters
        ----------
        None.


        Returns
        -------
        num_batches_per_epoch : int
            Number of batches per epoch.
        
        """

        return int(np.ceil(len(self.dataframe) / self.batch_size))


    def __getitem__(self, index) -> tuple[np.ndarray, np.ndarray]:

        """

        Method to get a batch of video frames and labels.


        Parameters
        ----------
        index : int
            Index of the batch.

        
        Returns
        -------
        batch_video_frames : numpy.ndarray
            Batch of video frames.

        batch_labels : numpy.ndarray
            Batch of labels.

        """

        # Get the indices for the current batch
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        batch_indices = self.indexes[start_index:end_index]


        # Initialize the lists for the current batch
        batch_video_frames = []
        batch_labels = []

        # Load video frames and labels for the current batch
        for i in batch_indices:

            video_path = self.dataframe.iloc[i, 0] 
            video_frames = videotoarray(video_path, self.dtype)
            
            labels = self.dataframe.iloc[i, 1:].astype(self.dtype)

            batch_video_frames.append(video_frames)
            batch_labels.append(labels)

        # Convert the lists to NumPy arrays
        batch_video_frames = np.array(batch_video_frames)
        batch_labels = np.array(batch_labels)


        return batch_video_frames, batch_labels


    def on_epoch_end(self) -> None:

        """
        
        Method to shuffle the indices after each epoch.


        Parameters
        ----------
        None.


        Returns
        -------
        None.

        """

        if self.shuffle:
            np.random.shuffle(self.indexes)
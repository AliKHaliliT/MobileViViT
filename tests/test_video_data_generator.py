import unittest
from MobileViViT.assets.utils.video_data_generator import VideoDataGenerator
import pandas as pd


class TestVideoDataGenerator(unittest.TestCase):

    def test_dataframe_wrong__type_type__error(self):

        # Arrange
        dataframe = None

        # Act and Assert
        with self.assertRaises(TypeError):
            VideoDataGenerator(dataframe=dataframe, batch_size=1)


    def test_batch__size__wrong_type__type_error(self):

        # Arrange
        dataframe = pd.DataFrame({"Address + FileName": ["util_resources/test_video.mp4", 
                                                         "util_resources/test_video.mp4", 
                                                         "util_resources/test_video.mp4"], 
                                  '0': [0, 1, 0], 
                                  '1': [1, 0, 1]})
        batch_size = None

        # Act and Assert
        with self.assertRaises(TypeError):
            VideoDataGenerator(dataframe=dataframe, batch_size=batch_size)            


    def test_batch__size_wrong__value_value__error(self):

        # Arrange
        dataframe = pd.DataFrame({"Address + FileName": ["util_resources/test_video.mp4", 
                                                         "util_resources/test_video.mp4", 
                                                         "util_resources/test_video.mp4"], 
                                  '0': [0, 1, 0], 
                                  '1': [1, 0, 1]})
        batch_size = -1

        # Act and Assert
        with self.assertRaises(ValueError):
            VideoDataGenerator(dataframe=dataframe, batch_size=batch_size)


    def test_shuffle_wrong__type_type__error(self):
        
        # Arrange
        dataframe = pd.DataFrame({"Address + FileName": ["util_resources/test_video.mp4", 
                                                         "util_resources/test_video.mp4", 
                                                         "util_resources/test_video.mp4"], 
                                  '0': [0, 1, 0], 
                                  '1': [1, 0, 1]})
        shuffle = None

        # Act and Assert
        with self.assertRaises(TypeError):
            VideoDataGenerator(dataframe=dataframe, batch_size=1, shuffle=shuffle)


    def test_normalization__value_wrong__type_type__error(self):
        
        # Arrange
        dataframe = pd.DataFrame({"Address + FileName": ["util_resources/test_video.mp4", 
                                                         "util_resources/test_video.mp4", 
                                                         "util_resources/test_video.mp4"], 
                                  '0': [0, 1, 0], 
                                  '1': [1, 0, 1]})
        normalization_value = "test"

        # Act and Assert
        with self.assertRaises(TypeError):
            VideoDataGenerator(dataframe=dataframe, batch_size=1, normalization_value=normalization_value)


    def test_path__col_wrong__type_type__error(self):
        
        # Arrange
        dataframe = pd.DataFrame({"Address + FileName": ["util_resources/test_video.mp4", 
                                                         "util_resources/test_video.mp4", 
                                                         "util_resources/test_video.mp4"], 
                                  '0': [0, 1, 0], 
                                  '1': [1, 0, 1]})
        path_col = -1.0

        # Act and Assert
        with self.assertRaises(TypeError):
            VideoDataGenerator(dataframe=dataframe, batch_size=1, path_col=path_col)


    def test_dtype_wrong__type_type__error(self):

        # Arrange
        dataframe = pd.DataFrame({"Address + FileName": ["util_resources/test_video.mp4", 
                                                         "util_resources/test_video.mp4", 
                                                         "util_resources/test_video.mp4"], 
                                  '0': [0, 1, 0], 
                                  '1': [1, 0, 1]})
        dtype = 1

        # Act and Assert
        with self.assertRaises(TypeError):
            VideoDataGenerator(dataframe=dataframe, batch_size=1, dtype=dtype)


    def test_output_dataframe_dataframe(self):

        # Arrange
        dataframe = pd.DataFrame({"Address + FileName": ["util_resources/test_video.mp4", 
                                                         "util_resources/test_video.mp4", 
                                                         "util_resources/test_video.mp4"], 
                                  '0': [0, 1, 0], 
                                  '1': [1, 0, 1]})
        
        # Act
        output = VideoDataGenerator(dataframe=dataframe, batch_size=2)

        # Assert
        self.assertEqual(output[0][1].shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
import unittest
from MobileViViT.assets.utils.video_file_to_numpy_array import video_file_to_numpy_array as videotoarray
import numpy as np


class TestVideoFileToNumpyArray(unittest.TestCase):

    def test_video__path_wrong__type_type__error(self):

        # Arrange
        video_path = None

        # Act and Assert
        with self.assertRaises(TypeError):
            videotoarray(video_path=video_path)


    def test_dtype_wrong__type_type__error(self):

        # Arrange
        video_path = "util_resources/video_file_to_numpy_array_test.mp4"
        dtype = 1

        # Act and Assert
        with self.assertRaises(TypeError):
            videotoarray(video_path=video_path, dtype=dtype)


    def test_video__read_not__opened_value__error(self):

        # Arrange
        video_path = "util_resources/video_file_to_numpy_array_test.mp"

        # Act and Assert
        with self.assertRaises(ValueError):
            videotoarray(video_path=video_path)


    def test_output_video__path_numpy__array(self):

        # Arrange
        video_path = "util_resources/video_file_to_numpy_array_test.mp4"
        
        # Act
        output = videotoarray(video_path=video_path)

        # Assert
        self.assertTrue(isinstance(output, np.ndarray))


if __name__ == "__main__":
    unittest.main()
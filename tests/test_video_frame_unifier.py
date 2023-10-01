import unittest
from MobileViViT.assets.utils.video_frame_unifier import video_frame_unifier
import os


class TestVideoFrameUnifier(unittest.TestCase):

    def test_input__path_wrong__type_type__error(self):

        # Arrange
        input_path = None
        output_path = "util_resources/test_output_path"

        # Act and Assert
        with self.assertRaises(TypeError):
            video_frame_unifier(input_path=input_path, output_path=output_path)
    

    def test_input__path_wrong__value_value__error(self):

        # Arrange
        input_path = "util_resources/test_video.mp4"
        output_path = "util_resources/test_output_path"

        # Act and Assert
        with self.assertRaises(ValueError):
            video_frame_unifier(input_path=input_path, output_path=output_path)


    def test_output__path_wrong__type_type__error(self):

        # Arrange
        input_path = "util_resources"
        output_path = None

        # Act and Assert
        with self.assertRaises(TypeError):
            video_frame_unifier(input_path=input_path, output_path=output_path)


    def test_output_input__path__and__output__path_not__empty__output__folder(self):

        # Arrange
        input_path = "util_resources"
        output_path = "util_resources/test_output_path"

        # Act
        video_frame_unifier(input_path=input_path, output_path=output_path)

        # Assert
        self.assertTrue(len(os.listdir(output_path)))


if __name__ == "__main__":
    unittest.main()
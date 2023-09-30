import cv2
import numpy as np


def video_file_to_numpy_array(video_path: str, dtype: np.dtype = np.float32) -> np.ndarray:

    """
    
    This method takes a video path and returns a numpy array of the video.


    Parameters
    ----------
    video_path : str
        Path of the video.

    dtype: numpy.dtype, optional
        Data type of the numpy array. The default value is np.float32.
        

    Returns
    -------
    video_array : numpy.ndarray
        Numpy array of the video.
    
    """

    if not isinstance(video_path, str):
        raise TypeError("video_path must be a string")
    if dtype is None:
        raise TypeError("dtype must be a numpy.dtype")
    elif not np.issubdtype(dtype, np.generic):
        raise TypeError("dtype must be a numpy.dtype")
    

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video")


    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_array = np.empty((frame_count, frame_height, frame_width, 3), dtype=dtype)


    for index in range(frame_count):
        ret, video_array[index] = cap.read()
        if not ret:
            break


    cap.release()


    return video_array
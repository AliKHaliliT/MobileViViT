import os
import cv2
import numpy as np
import shutil


def video_frame_unifier(input_path: str, output_path: str) -> None:

    """

    Unifies the number of frames of video files based on the
    number of frames of the video with the most frames in a folder
    with zero padding. Padded video files will have the same name as the original video files
    with the suffix "_padded" added to the end of the file name. The longest video file
    will not be padded and will be copied to the output folder as is with the same name.


    Parameters
    ----------
    input_path : str
        Input path to the folder containing the video files.

    output_path : str
        Output path to the folder where the padded video files will be saved.


    Returns
    -------
    None.

    """

    if not isinstance(input_path, str):
        raise TypeError("input_path must be a string")
    if not os.path.isdir(input_path):
        raise ValueError("input_path must be a valid path to a folder")
    if not isinstance(output_path, str):
        raise TypeError("output_path must be a string")
    

    if not os.path.isdir(output_path):
        os.makedirs(output_path)


    # Get the paths of the files in the folder
    video_files = [filename for filename in os.listdir(input_path)]


    # Find the video with the most frames
    max_frames = 0

    for video_file in video_files:

        try:
            cap = cv2.VideoCapture(os.path.join(input_path, video_file))
        except:

            print(f"Could not read {os.path.join(input_path, video_file)}.")
            video_files.remove(video_file)
            continue

        if not cap.isOpened():

            print(f"Could not open the video at {os.path.join(input_path, video_file)}")
            video_files.remove(video_file)
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count > max_frames:
            max_frames = frame_count

        cap.release()


    # Pad other videos to match the maximum frame count
    for video_file in video_files:

        cap = cv2.VideoCapture(os.path.join(input_path, video_file))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count < max_frames:

            # Calculate the number of frames to pad
            frames_to_pad = max_frames - frame_count
            
            # Read the first frame to get dimensions
            ret, frame = cap.read()
            height, width, channels = frame.shape

            if ret:
                
                # Create a VideoWriter to save the padded video
                output_file = os.path.join(output_path, os.path.splitext(video_file)[0] + 
                                            "_padded" + 
                                            os.path.splitext(video_file)[1])
                out = cv2.VideoWriter(output_file, 
                                        cv2.VideoWriter_fourcc(*'mp4v'), 
                                        cap.get(cv2.CAP_PROP_FPS), 
                                        (width, height))
                
                # Write the original frames
                while ret:
                    
                    out.write(frame)
                    ret, frame = cap.read()
                
                # Pad with zeros
                for _ in range(frames_to_pad):
                    out.write(np.zeros((height, width, channels), dtype=np.uint8))
                    
                out.release()
        
        else: 
            # Copy the video file if it has the maximum number of frames
            # without padding
            shutil.copyfile(os.path.join(input_path, video_file), 
                            os.path.join(output_path, video_file))
            
        
        cap.release()
# Disable logs to reduce clutter
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


import pandas as pd
from MobileViViT.assets.utils.video_data_generator import VideoDataGenerator
from MobileViViT import MobileViViTXXS


# Config
num_output_units = 2
batch_size = 1
epochs = 1


# Sample input video
path_to_video = "util_resources/test_video.mp4"
video_data = pd.DataFrame({
    "Address + FileName": [path_to_video],
    "0": [0],
    "1": [1]
})


# Data generator
data_generator = VideoDataGenerator(dataframe=video_data, batch_size=batch_size)


# Initialize and train model
model = MobileViViTXXS(num_output_units=num_output_units)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(data_generator, epochs=epochs)

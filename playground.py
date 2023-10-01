# Disable logs to reduce clutter
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


import pandas as pd
from MobileViViT.assets.utils.video_data_generator import VideoDataGenerator
from MobileViViT import MobileViViTXXS


num_output_units = 2
batch_size = 1
epochs = 1


path_to_sample_video = "util_resources/test_video.mp4"

video_data = pd.DataFrame({"Address + FileName": [path_to_sample_video], 
                           '0': [0], 
                           '1': [1]})

data_generator = VideoDataGenerator(dataframe=video_data, batch_size=batch_size, normalization_value=255)


mobilevivit_xss = MobileViViTXXS(num_output_units=num_output_units)

mobilevivit_xss.compile(optimizer="adam", 
                        loss="categorical_crossentropy", 
                        metrics=["accuracy"])

mobilevivit_xss.fit(data_generator, epochs=epochs)
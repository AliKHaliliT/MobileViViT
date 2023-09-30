# MobileViViT

![tests](https://github.com/AliKHaliliT/MobileViViT/actions/workflows/tests.yml/badge.svg)


This repository contains the implementation of MobileViViT (Mobile Video Vision Transformers), which is an adaptation of [MobileViT](https://arxiv.org/abs/2110.02178) designed explicitly for higher dimensional tasks. <b> It is essential to note that this is not a research-based effort but rather an adaptation of [MobileViT](https://arxiv.org/abs/2110.02178). </b> However, utmost care has been taken to maintain the original integrity of MobileViT as much as possible.

The MobileViViT variants implemented in this repository are:

- MobileViViT-S
- MobileViViT-XS
- MobileViViT-XXS

In addition, the MobileViViT has been designed with separate building blocks, which can be used independently or together to construct other components. Furthermore, any necessary utility module that may be required for the training and evaluation of the models has also been created as separate modules (e.g., Video Data Generator, Custom Training Loop) and can be found in the ```utils``` directory. Also, each module includes documentation that provides an in-depth understanding of its implementation details.

Furthermore, this repository includes the complete or partial implementation of the following research papers, which were either directly utilized or served as sources of inspiration:

- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187)
- [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
- [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681)
- [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [ViViT: A Video Vision Transformer](https://arxiv.org/abs/2103.15691)
- [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178)

This repository's materials and references have been appropriately cited and acknowledged. No attempt has been made to claim any of the work as original. If you discover any materials in this repository that violate any terms of use, please contact me, and I will take the necessary steps to rectify the issue.

## Installation
The repository was created using Python version ```3.11.xx``` along with Tensorflow and Keras. If you have the correct Python version installed, you can install the required dependencies using the following command:

```
pip install -r requirements.txt
```

Please be aware that the listed dependencies may not cover all requirements and you might need to install additional packages depending on your system configuration to satisfy the possible depencency requirements of the packages listed in the ```requirements.txt``` file.

## Package Structure
```
📦 MobileViViT/
 ┣ 📜 __init__.py
 ┗ 📂 assets/
     ┣ 📜 __init__.py
     ┣ 📂 activations/
     ┃  ┣ 📜 __init__.py
     ┃  ┣ 📜 hard_swish.py
     ┃  ┣ 📜 mish.py
     ┃  ┗ 📜 sine.py
     ┣ 📂 blocks/
     ┃  ┣ 📜 __init__.py
     ┃  ┣ 📜 mlp.py
     ┃  ┣ 📜 mobilevivit.py
     ┃  ┣ 📜 mvnblock.py
     ┃  ┣ 📜 siren.py
     ┃  ┗ 📜 transformer.py
     ┣ 📂 layers/
     ┃  ┣ 📜 __init__.py
     ┃  ┣ 📜 conv2plus1d.py
     ┃  ┣ 📜 conv_layer.py
     ┃  ┣ 📜 fc_layer.py
     ┃  ┣ 📜 fold.py
     ┃  ┣ 📜 positional_encoder.py
     ┃  ┣ 📜 sae_3d.py
     ┃  ┣ 📜 sine_layer.py
     ┃  ┣ 📜 ssae_3d.py
     ┃  ┣ 📜 transformer_layer.py
     ┃  ┣ 📜 tubelet_embedding.py
     ┃  ┗ 📜 unfold.py
     ┣ 📂 utils/
     ┃  ┣ 📜 __init__.py
     ┃  ┣ 📜 activation_function.py
     ┃  ┣ 📜 low_resource_training_scheme.py
     ┃  ┣ 📜 move_column_to_the_beginning.py
     ┃  ┣ 📜 progress_bar.py
     ┃  ┣ 📜 sine_layer_initializer.py
     ┃  ┣ 📜 squeeze_and_excitation.py
     ┃  ┣ 📜 video_data_generator.py
     ┃  ┗ 📜 video_file_to_numpy_array.py
 ┣ 📜 mobilevivit_s.py
 ┣ 📜 mobilevivit_xs.py
 ┗ 📜 mobilevivit_xxs.py
```

## Usage
The MobileViViT models can be imported and used as follows:

```python
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

data_generator = VideoDataGenerator(dataframe=video_data, batch_size=batch_size)


mobilevivit_xss = MobileViViTXXS(num_output_units=num_output_units)

mobilevivit_xss.compile(optimizer="adam", 
                        loss="categorical_crossentropy", 
                        metrics=["accuracy"])

mobilevivit_xss.fit(data_generator, epochs=epochs)
```

## License
This work is under an [MIT](https://choosealicense.com/licenses/mit/) License.
# MobileViViT
<div style="display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 10px;">
    <img src="https://img.shields.io/github/license/AliKHaliliT/MobileViViT" alt="License">
    <img src="https://github.com/AliKHaliliT/MobileViViT/actions/workflows/tests.yml/badge.svg" alt="tests">
    <img src="https://img.shields.io/github/last-commit/AliKHaliliT/MobileViViT" alt="Last Commit">
    <img src="https://img.shields.io/github/issues/AliKHaliliT/MobileViViT" alt="Open Issues">
</div>
<br/>

This repository provides an implementation of **MobileViViT (Mobile Video Vision Transformers)** — an adaptation of [MobileViT](https://arxiv.org/abs/2110.02178) designed for higher-dimensional tasks such as video.

> ⚠️ **Note**: This is not a research-based project. It is an adaptation of MobileViT with careful consideration to preserve the integrity of the original work.

---

## Available Variants

* **MobileViViT-S**
* **MobileViViT-XS**
* **MobileViViT-XXS**

---

## Design Philosophy

The implementation is built from modular components that can be reused independently or combined to construct new architectures. Utility modules (e.g., custom training loops, video data generators) are included under `utils/` and come with documentation for easier understanding and extension.

---

## References & Inspirations

This repository includes complete or partial implementations inspired by or directly adapted from the following works:

* [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
* [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248)
* [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
* [Bag of Tricks for Image Classification with CNNs](https://arxiv.org/abs/1812.01187)
* [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
* [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681)
* [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)
* [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
* [ViViT: A Video Vision Transformer](https://arxiv.org/abs/2103.15691)
* [MobileViT](https://arxiv.org/abs/2110.02178)

All references have been properly acknowledged. If any material violates terms of use, please contact me, and I will promptly address it.

---

## Quick Start

The repository was developed with **Python 3.11.x**, **TensorFlow**, and **Keras**.

Clone the repository and install dependencies:

```bash
git clone https://github.com/AliKHaliliT/MobileViViT.git
cd MobileViViT
pip install -r requirements.txt
```

(Optional) Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

> ⚠️ Depending on your system, you may need to install additional packages beyond those listed in `requirements.txt`.

---

## Package Structure

```
MobileViViT/
 ├── __init__.py
 ├── assets/
 │   ├── __init__.py
 │   ├── activations/
 │   │   ├── __init__.py
 │   │   ├── hard_swish.py
 │   │   ├── mish.py
 │   │   └── sine.py
 │   ├── blocks/
 │   │   ├── __init__.py
 │   │   ├── mlp.py
 │   │   ├── mobilevivit.py
 │   │   ├── mvnblock.py
 │   │   ├── siren.py
 │   │   └── transformer.py
 │   ├── layers/
 │   │   ├── __init__.py
 │   │   ├── conv2plus1d.py
 │   │   ├── conv_layer.py
 │   │   ├── fc_layer.py
 │   │   ├── fold.py
 │   │   ├── positional_encoder.py
 │   │   ├── sae_3d.py
 │   │   ├── sine_layer.py
 │   │   ├── ssae_3d.py
 │   │   ├── transformer_layer.py
 │   │   ├── tubelet_embedding.py
 │   │   └── unfold.py
 │   └── utils/
 │       ├── __init__.py
 │       ├── activation_function.py
 │       ├── low_resource_training_scheme.py
 │       ├── move_column_to_the_beginning.py
 │       ├── progress_bar.py
 │       ├── sine_layer_initializer.py
 │       ├── squeeze_and_excitation.py
 │       ├── video_data_generator.py
 │       ├── video_file_to_numpy_array.py
 │       └── video_frame_unifier.py
 ├── mobilevivit_s.py
 ├── mobilevivit_xs.py
 └── mobilevivit_xxs.py
```

---

## Usage Example

```python
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
```

---

## License

This work is under an [MIT](https://choosealicense.com/licenses/mit/) License.
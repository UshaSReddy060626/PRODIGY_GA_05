"""
utils.py
Shared helper functions for loading, preprocessing, and saving images
used by both the fast (TF-Hub) and classic (VGG19/Gatys) style transfer
implementations.
"""

import os
import numpy as np
import tensorflow as tf
import PIL.Image


def load_img(path_to_img, max_dim=512):
    """Load an image from disk, resize it so its longest side is
    `max_dim` pixels, and return it as a float32 tensor of shape
    (1, H, W, 3) with pixel values in [0, 1].
    """
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def tensor_to_image(tensor):
    """Convert a (1, H, W, 3) float tensor in [0, 1] (or [0, 255]) back
    into a PIL Image.
    """
    tensor = np.array(tensor)
    if tensor.max() <= 1.0:
        tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def save_image(tensor, out_path):
    """Save a stylized tensor to disk, creating parent dirs if needed."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img = tensor_to_image(tensor)
    img.save(out_path)
    print(f"Saved stylized image to: {out_path}")
    return out_path

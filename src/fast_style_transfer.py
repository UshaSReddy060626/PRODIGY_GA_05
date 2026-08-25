"""
fast_style_transfer.py

Quick style transfer using Google Magenta's pretrained "arbitrary image
stylization" model from TensorFlow Hub. This gives a result in a couple
of seconds (no per-image optimization loop required).

Usage:
    python src/fast_style_transfer.py \
        --content examples/content/your_photo.jpg \
        --style examples/style/your_painting.jpg \
        --output outputs/fast_result.jpg
"""

import argparse
import tensorflow as tf
import tensorflow_hub as hub

from utils import load_img, save_image

MODEL_URL = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"


def run_fast_style_transfer(content_path, style_path, output_path, max_dim=512):
    print("Loading pretrained stylization model from TF-Hub...")
    hub_model = hub.load(MODEL_URL)

    print(f"Loading content image: {content_path}")
    content_image = load_img(content_path, max_dim=max_dim)

    print(f"Loading style image: {style_path}")
    style_image = load_img(style_path, max_dim=max_dim)

    print("Running stylization...")
    outputs = hub_model(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]

    return save_image(stylized_image, output_path)


def main():
    parser = argparse.ArgumentParser(description="Fast neural style transfer (TF-Hub)")
    parser.add_argument("--content", required=True, help="Path to content image")
    parser.add_argument("--style", required=True, help="Path to style image")
    parser.add_argument("--output", default="outputs/fast_result.jpg", help="Path to save result")
    parser.add_argument("--max-dim", type=int, default=512, help="Max image dimension")
    args = parser.parse_args()

    run_fast_style_transfer(args.content, args.style, args.output, args.max_dim)


if __name__ == "__main__":
    main()

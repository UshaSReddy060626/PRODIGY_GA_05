"""
style_transfer_vgg19.py

Classic Neural Style Transfer (Gatys et al., 2015) implemented with a
pretrained VGG19 network as a feature extractor. Unlike the fast
TF-Hub version, this actually runs a gradient-descent optimization
loop directly on the pixels of a generated image, so it's slower but
fully customizable (content/style weight balance, choice of layers,
number of steps, etc).

Usage:
    python src/style_transfer_vgg19.py \
        --content examples/content/your_photo.jpg \
        --style examples/style/your_painting.jpg \
        --output outputs/vgg19_result.jpg \
        --epochs 10 --steps-per-epoch 100
"""

import argparse
import time

import tensorflow as tf

from utils import load_img, save_image

# Layers used to represent style (multiple layers -> texture at
# multiple scales) and content (a single deep layer -> preserves
# high-level structure but not exact pixels).
CONTENT_LAYERS = ["block5_conv2"]
STYLE_LAYERS = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]


def vgg_layers(layer_names):
    """Build a model that returns the intermediate activations of the
    requested VGG19 layers."""
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    return tf.keras.Model([vgg.input], outputs)


def gram_matrix(input_tensor):
    """Gram matrix = correlations between feature channels -> captures
    'style' (textures, colors) while discarding spatial arrangement."""
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super().__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        # Expects float input in [0, 1]
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (
            outputs[: self.num_style_layers],
            outputs[self.num_style_layers :],
        )
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {
            name: value for name, value in zip(self.content_layers, content_outputs)
        }
        style_dict = {
            name: value for name, value in zip(self.style_layers, style_outputs)
        }
        return {"content": content_dict, "style": style_dict}


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def style_content_loss(outputs, style_targets, content_targets,
                        style_weight, content_weight,
                        num_style_layers, num_content_layers):
    style_outputs = outputs["style"]
    content_outputs = outputs["content"]

    style_loss = tf.add_n(
        [
            tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
            for name in style_outputs.keys()
        ]
    )
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n(
        [
            tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
            for name in content_outputs.keys()
        ]
    )
    content_loss *= content_weight / num_content_layers

    return style_loss + content_loss


def run_style_transfer(
    content_path,
    style_path,
    output_path,
    epochs=10,
    steps_per_epoch=100,
    style_weight=1e-2,
    content_weight=1e4,
    total_variation_weight=30,
    learning_rate=0.02,
    max_dim=512,
):
    content_image = load_img(content_path, max_dim=max_dim)
    style_image = load_img(style_path, max_dim=max_dim)

    extractor = StyleContentModel(STYLE_LAYERS, CONTENT_LAYERS)
    style_targets = extractor(style_image)["style"]
    content_targets = extractor(content_image)["content"]

    image = tf.Variable(content_image)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.99, epsilon=1e-1)

    num_style_layers = len(STYLE_LAYERS)
    num_content_layers = len(CONTENT_LAYERS)

    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(
                outputs,
                style_targets,
                content_targets,
                style_weight,
                content_weight,
                num_style_layers,
                num_content_layers,
            )
            loss += total_variation_weight * tf.reduce_sum(tf.image.total_variation(image))

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))
        return loss

    print(f"Running {epochs} epochs x {steps_per_epoch} steps "
          f"({epochs * steps_per_epoch} total optimization steps)...")
    start = time.time()
    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            loss = train_step(image)
        print(f"Epoch {n + 1}/{epochs} complete — step {step}, loss={float(loss):.1f}")

    print(f"Total time: {time.time() - start:.1f}s")
    return save_image(image, output_path)


def main():
    parser = argparse.ArgumentParser(description="Classic neural style transfer (VGG19 / Gatys et al.)")
    parser.add_argument("--content", required=True, help="Path to content image")
    parser.add_argument("--style", required=True, help="Path to style image")
    parser.add_argument("--output", default="outputs/vgg19_result.jpg", help="Path to save result")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--style-weight", type=float, default=1e-2)
    parser.add_argument("--content-weight", type=float, default=1e4)
    parser.add_argument("--tv-weight", type=float, default=30, help="Total variation weight (smoothness)")
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--max-dim", type=int, default=512)
    args = parser.parse_args()

    run_style_transfer(
        args.content,
        args.style,
        args.output,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        style_weight=args.style_weight,
        content_weight=args.content_weight,
        total_variation_weight=args.tv_weight,
        learning_rate=args.lr,
        max_dim=args.max_dim,
    )


if __name__ == "__main__":
    main()

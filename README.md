# 🎨 Neural Style Transfer

> Transform an ordinary photograph into an artistic masterpiece using Deep Learning.

Neural Style Transfer (NST) is a deep learning technique that combines the **content of one image** with the **artistic style of another image**.

This project implements **two different approaches** to Neural Style Transfer:

| Approach                | Implementation                          |   Speed | Best For                  |
| ----------------------- | --------------------------------------- | ------: | ------------------------- |
| ⚡ Fast Style Transfer   | Google Magenta model via TensorFlow Hub | Seconds | Quick demonstrations      |
| 🎨 VGG19 Style Transfer | Gatys et al. optimization               | Minutes | Quality and customization |

The project demonstrates both **pretrained inference-based stylization** and the **classic optimization-based approach** to understand how Neural Style Transfer works at a deeper level.

---

## ✨ What is Neural Style Transfer?

Neural Style Transfer separates an image into two concepts:

* **Content** → The objects, shapes, structure, and spatial arrangement of the image.
* **Style** → Colors, textures, patterns, brush strokes, and artistic appearance.

A pretrained CNN such as **VGG19** is used to extract these representations.

The goal is to generate a new image that:

> Preserves the content of the content image while reproducing the visual style of the style image.

For example:

```text
Content Image + Style Image
          ↓
   Neural Style Transfer
          ↓
     Stylized Image
```

---

## 🧠 How Neural Style Transfer Works

### 1. Content Representation

Deep layers of a CNN capture high-level information such as:

* Objects
* Shapes
* Structure
* Spatial arrangement

The generated image is optimized to remain similar to the content image in these deeper feature representations.

### 2. Style Representation

Style is represented using the relationships between different feature maps.

These relationships are captured using a **Gram Matrix**.

The Gram matrix helps represent:

* Textures
* Colors
* Patterns
* Brush-stroke-like structures

Style information is extracted from multiple CNN layers.

### 3. Optimization

In the classic VGG19 approach, the generated image itself is optimized.

The optimization attempts to minimize:

```text
Total Loss
    =
Content Loss
+
Style Loss
+
Total Variation Loss
```

### Content Loss

Measures how different the generated image is from the original content image.

```text
Content Loss = Difference between content features
```

### Style Loss

Measures the difference between the style representation of the generated image and the style image.

```text
Style Loss = Difference between Gram Matrices
```

### Total Variation Loss

Encourages smoothness and reduces unwanted noise.

```text
TV Loss → smoother and cleaner image
```

---

# ⚡ Approach 1 — Fast Style Transfer

The fast implementation uses Google's pretrained:

```text
arbitrary-image-stylization-v1-256
```

model through **TensorFlow Hub**.

Unlike the classic approach, the image does not need to be optimized for hundreds of iterations.

Instead:

```text
Content Image
      +
Style Image
      ↓
Pretrained Model
      ↓
Stylized Image
```

### Advantages

* Very fast
* One forward pass
* Easy to use
* Suitable for demonstrations
* Good for real-time or near-real-time applications

### Limitations

* Less control over optimization
* Output quality depends on the pretrained model
* Can produce blocky or noisy results on complex images

---

# 🎨 Approach 2 — Classic VGG19 NST

The second implementation follows the classic approach introduced by:

**Gatys, Ecker & Bethge — A Neural Algorithm of Artistic Style**

A pretrained **VGG19** network is used as a feature extractor.

The generated image starts from the content image and is iteratively optimized.

```text
Content Image
      ↓
Initial Generated Image
      ↓
VGG19 Feature Extraction
      ↓
Calculate Content Loss
      +
Calculate Style Loss
      +
Calculate TV Loss
      ↓
Gradient Descent
      ↓
Updated Image
      ↓
Repeat
      ↓
Final Stylized Image
```

### Advantages

* Highly customizable
* Better control over style strength
* Can produce high-quality artistic results
* Excellent for understanding the theory behind NST

### Limitations

* Computationally expensive
* Requires many optimization steps
* Slower than pretrained feed-forward models

---

# 🔬 Fast vs Classic NST

| Feature            | Fast Style Transfer          | VGG19 Optimization |
| ------------------ | ---------------------------- | ------------------ |
| Model              | Pretrained stylization model | VGG19              |
| Optimization       | ❌ No                         | ✅ Yes              |
| Speed              | ⚡ Seconds                    | 🐢 Minutes         |
| Style flexibility  | High                         | High               |
| Fine control       | Limited                      | Excellent          |
| Computational cost | Low                          | High               |
| Best for           | Quick results                | Experimentation    |
| Output tuning      | Limited                      | Extensive          |

---

# 📁 Project Structure

```text
neural-style-transfer/
│
├── src/
│   ├── fast_style_transfer.py
│   ├── style_transfer_vgg19.py
│   └── utils.py
│
├── examples/
│   ├── content/
│   │   └── Landscape.jpg
│   │
│   └── style/
│       └── royal_evening.jpg
│
├── outputs/
│   ├── fast_result.jpg
│   └── vgg19_result.jpg
│
├── requirements.txt
└── README.md
```

---

# 🛠️ Technologies Used

* **Python**
* **TensorFlow**
* **TensorFlow Hub**
* **VGG19**
* **NumPy**
* **Pillow**
* **Deep Learning**
* **Convolutional Neural Networks**
* **Gradient Descent**
* **Transfer Learning**

---

# ⚙️ Installation

## Windows — PowerShell

```powershell
git clone https://github.com/<your-username>/neural-style-transfer.git
cd neural-style-transfer
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## macOS / Linux

```bash
git clone https://github.com/<your-username>/neural-style-transfer.git
cd neural-style-transfer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

# 🚀 Usage

Place your images inside:

```text
examples/content/
examples/style/
```

For example:

```text
examples/
├── content/
│   └── Landscape.jpg
│
└── style/
    └── royal_evening.jpg
```

---

## ⚡ Fast Style Transfer

```powershell
python src\fast_style_transfer.py --content examples\content\Landscape.jpg --style examples\style\royal_evening.jpg --output outputs\fast_result.jpg
```

This approach generates a stylized image in a few seconds on a typical machine.

---

## 🎨 Classic VGG19 Style Transfer

```powershell
python src\style_transfer_vgg19.py --content examples\content\Landscape.jpg --style examples\style\royal_evening.jpg --output outputs\vgg19_result.jpg
```

The VGG19 implementation performs iterative optimization and therefore takes considerably longer.

---

# 🎛️ VGG19 Parameters

The classic implementation provides several parameters for controlling the generated image.

| Parameter           | Default | Description                                             |
| ------------------- | ------: | ------------------------------------------------------- |
| `--style-weight`    |  `1e-2` | Controls the strength of the artistic style             |
| `--content-weight`  |   `1e4` | Controls how strongly the original content is preserved |
| `--tv-weight`       |    `30` | Controls image smoothness                               |
| `--epochs`          |    `10` | Number of optimization epochs                           |
| `--steps-per-epoch` |   `100` | Optimization steps per epoch                            |
| `--max-dim`         |   `512` | Maximum image dimension                                 |

### Increasing Style Weight

```text
Higher style weight
        ↓
Stronger artistic effect
        ↓
Less content preservation
```

### Increasing Content Weight

```text
Higher content weight
        ↓
Stronger preservation of original image
        ↓
Weaker artistic transformation
```

### Increasing TV Weight

```text
Higher TV weight
        ↓
Smoother image
        ↓
Less visual noise
```

---

# 🧪 Quick VGG19 Experiment

The full VGG19 optimization can take several minutes.

For a quick test, use:

```powershell
python src\style_transfer_vgg19.py --content examples\content\Landscape.jpg --style examples\style\royal_evening.jpg --output outputs\vgg19_preview.jpg --epochs 2 --steps-per-epoch 50
```

This performs:

```text
2 epochs × 50 steps = 100 optimization steps
```

instead of the default:

```text
10 epochs × 100 steps = 1000 optimization steps
```

The preview is useful for verifying that everything works before running the complete optimization.

---

# 🖼️ Example

### Content

A beach cove containing:

* Cliffs
* Rocks
* Shells
* Natural landscape structure

### Style

An Indian miniature-style painting depicting a woman on a balcony at sunset.

### Expected Result

The generated image preserves the **structure and layout of the beach scene** while incorporating the painting's:

* Colors
* Textures
* Patterns
* Artistic appearance

Example output:

```text
Content Image
      +
Style Image
      ↓
Stylized Beach Scene
```

The VGG19 implementation generally produces a smoother and more painterly result, while the fast model produces results much more quickly.

---

# 📊 What This Project Demonstrates

This project provides practical experience with:

* Convolutional Neural Networks
* Transfer Learning
* Feature Extraction
* VGG19
* TensorFlow
* TensorFlow Hub
* Gram Matrices
* Content Loss
* Style Loss
* Total Variation Loss
* Gradient Descent
* Image Optimization
* Pretrained Deep Learning Models
* Computer Vision

---

# 💡 Key Learning

Neural Style Transfer demonstrates an important idea in Deep Learning:

> A neural network does not only learn to classify images. Its internal feature representations can also be used to manipulate and generate visual information.

The project also highlights the difference between:

```text
Optimization-based Deep Learning
            vs
Pretrained Feed-Forward Inference
```

The classic method provides more control but requires significant computation, while the pretrained approach sacrifices some flexibility for dramatically faster inference.

---

# 🔮 Possible Improvements

Future improvements could include:

* [ ] Add a web interface using Flask
* [ ] Add a Streamlit interface
* [ ] Support batch image processing
* [ ] Add GPU acceleration
* [ ] Allow users to control style strength interactively
* [ ] Add image resizing and preprocessing options
* [ ] Compare different VGG19 layers
* [ ] Add multiple style images
* [ ] Implement video style transfer
* [ ] Build a real-time webcam stylization application
* [ ] Add automatic output quality comparison

---

# 📚 References

* [TensorFlow — Neural Style Transfer](https://www.tensorflow.org/tutorials/generative/style_transfer)
* [Gatys, Ecker & Bethge — A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
* [TensorFlow Hub — Arbitrary Image Stylization](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2)
* [How do Neural Style Transfers Work? — Towards Data Science](https://towardsdatascience.com/how-do-neural-style-transfers-work-b76de101eb3)
* [Neural Style Transfer with TensorFlow — GeeksforGeeks](https://www.geeksforgeeks.org/neural-style-transfer-with-tensorflow/)

---
# 🖼️ Image Credits & Attribution

The example content and style images used in this project were sourced from **Pinterest** for educational and demonstration purposes.

The images belong to their respective original creators and copyright holders. They are **not claimed as original work** by the author.

* Image source: [Pinterest](https://www.pinterest.com/)
* Images are used only to demonstrate the Neural Style Transfer technique.
* For redistribution or commercial use, please obtain permission from the respective copyright holders.
* If publishing the generated results publicly, credit the original creators/sources of the input images where possible.

> **Note:** The Neural Style Transfer implementation and code in this repository are the author's work; the example input images are externally sourced.


# ⭐ Conclusion

This project implements Neural Style Transfer using **two complementary approaches**.

The **TensorFlow Hub model** provides fast and practical stylization, while the **VGG19 optimization approach** demonstrates the underlying mathematics and deep learning concepts behind classic Neural Style Transfer.

Together, they provide both a practical implementation and a deeper understanding of how CNN feature representations can be used to transform images artistically.

---

## 👩‍💻 Author

**<Usha.S.Reddy>**

If you found this project useful, consider giving the repository a ⭐.

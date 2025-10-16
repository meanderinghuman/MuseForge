# 🏛️ MuseForge

<div align="center">

**Where Heritage Finds Its Voice.**

*An AI-powered storyteller that transforms silent cultural artifacts into vivid, authentic narratives, breathing life into India's rich and diverse heritage.*

</div>

<div align="center">

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
![GitHub stars](https://img.shields.io/github/stars/meanderinghuman/MuseForge?style=social)
![GitHub forks](https://img.shields.io/github/forks/meanderinghuman/MuseForge?style=social)
![GitHub issues](https://img.shields.io/github/issues/meanderinghuman/MuseForge)

[🚀 Get Started](#-getting-started) • [✨ Features](#-core-features) • [🏗️ Architecture](#-the-museforge-architecture) • [🤝 Contribute](#-join-the-forge)

---

</div>

## 🌟 The Vision: Preserving a Digital Soul

Countless cultural treasures exist only as silent images, their stories and contexts at risk of being lost to time. MuseForge addresses this critical gap by serving as a digital artisan, forging a connection between visual heritage and the rich narratives they hold.

Our mission is to:
-   📜 **Create Digital Immortality**: Transform visual heritage into enduring textual sagas, safeguarding them for future generations.
-   🌍 **Build a Global Knowledge Bridge**: Make India's profound cultural wisdom accessible to a global audience of researchers, students, and enthusiasts.
-   🎓 **Forge Engaging Educational Tools**: Create a dynamic new way to learn about history, art, and culture through AI-driven storytelling.

## ✨ Core Features

-   **Progressive Learning Pipeline**: A unique 3-stage process that teaches the AI to first **recognize** elements, then **understand** context, and finally **generate** rich narratives.
-   **Diverse Cultural Dataset**: Trained on **960 curated images** spanning 5 major geographical regions of India, ensuring broad and authentic representation.
-   **Efficient & Powerful**: Utilizes **LoRA fine-tuning** to achieve an **85% performance boost** (loss reduction from 130 to 19.7) on accessible hardware.
-   **Culturally-Aware Storytelling**: Generates historically and culturally accurate narratives that honor the source material.
-   **Innovative Methodology**: Introduces a novel approach to fine-tuning Vision-Language Models for specialized, high-context domains.

## 📊 Project at a Glance

| Metric                | Achievement                                |
| --------------------- | ------------------------------------------ |
| 🧠 **Model** | Qwen2-VL (2B) with LoRA Fine-Tuning        |
| 📷 **Dataset Size** | 960 Curated Cultural Images                |
| 📈 **Performance** | 85% Loss Reduction (130 → 19.7)            |
| 🗺️ **Coverage** | 5 Major Indian Geographical Regions        |
| 🏗️ **Methodology** | 3-Stage Progressive Learning               |
| ✅ **Output Quality** | High-Fidelity, Culturally Authentic Stories|

## 🏗️ The MuseForge Architecture

MuseForge employs a three-stage progressive learning pipeline that mimics a human expert's process: observing details, understanding patterns, and finally, telling a compelling story.

```mermaid
graph TD
    A[Cultural Image Input] --> B[👁️ Vision Encoder (Qwen2-VL)]
    B --> C[🧠 Language Model Core]
    C --> D[**Stage 1: The Observer**<br/>Identifies cultural elements]
    D --> E[**Stage 2: The Scholar**<br/>Understands regional patterns]
    E --> F[**Stage 3: The Storyteller**<br/>Weaves an authentic narrative]
    F --> G[📖 Forged Cultural Story]
```

### 🔍 Stage 1 — The Eye of the Observer

  - **Goal**: To see and identify key components.
  - **Learns**: To recognize temples, sculptures, traditional clothing, architectural features, and artifacts.
  - **Training**: 1 Epoch | Learning Rate: `2e-4`

### 🎨 Stage 2 — The Mind of the Scholar

  - **Goal**: To understand the relationships and context.
  - **Learns**: To connect elements to broader architectural styles (Dravidian, Mughal), regional patterns, and cultural significance.
  - **Training**: 1 Epoch | Learning Rate: `1e-4`

### 📚 Stage 3 — The Voice of the Storyteller

  - **Goal**: To synthesize knowledge into a flowing narrative.
  - **Learns**: To generate authentic, contextual stories that preserve historical accuracy and cultural nuance.
  - **Training**: 2 Epochs | Learning Rate: `5e-5`

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/meanderinghuman/MuseForge.git
cd MuseForge

# Install core dependencies
pip install -r requirements.txt

# Install specialized vision-language dependencies
pip install transformers[vision] torch torchvision peft accelerate bitsandbytes
```

### Forge Your First Story

```python
from inference import MuseForge

# Initialize the model from a trained checkpoint
story_forge = MuseForge("./models/cultural_vlm_trained")

# Select a cultural image
image_path = "examples/temple_image.jpg"

# Forge a narrative
story = story_forge.forge_story(image_path)
print(story)
```

### Example Output

```
This authentic East India scene while reveals a group of people walking across a bridge. The distinctive colonial calcutta showcases the region's unique architectural heritage and Regional cultural elements including poila boishakh, adda culture provide authentic local context. People adorned in traditional dress, ethnic wear embody the living culture....
```

## 📈 Performance & Results

The progressive approach shows a clear and dramatic improvement in the model's learning, with each stage building effectively on the last.

| Stage | Task                    | Loss Reduction | Key Learning                |
| ----- | ----------------------- | -------------- | --------------------------- |
| 1     | Element Recognition     | 504 → 217      | Artifact Identification     |
| 2     | Pattern Understanding   | 217 → 201      | Regional Style Comprehension|
| 3     | Story Generation        | 184 → **19.7** | Authentic Narrative Creation|

## 🛠️ Tech Stack

  - **Base Model**: `Qwen2-VL-2B-Instruct`
  - **Fine-tuning**: `LoRA` (r=8, α=32, 7 target modules)
  - **Quantization**: 4-bit with `bfloat16` precision for efficiency
  - **Frameworks**: PyTorch, Hugging Face `transformers`, `PEFT`, `accelerate`
  - **Optimization**: Gradient Checkpointing & Quantization
  - **Hardware**: 8 GB+ GPU Recommended (e.g., NVIDIA RTX 3060)

## 🤝 Join the Forge

We welcome contributions from cultural enthusiasts, AI researchers, and heritage preservationists. Let's build this together!

### How to Contribute

1.  🍴 **Fork** the repository.
2.  🌿 **Create** a new branch (`git checkout -b feature/your-amazing-feature`).
3.  💾 **Commit** your changes (`git commit -m 'Add your amazing feature'`).
4.  📤 **Push** to the branch (`git push origin feature/your-amazing-feature`).
5.  🔀 **Open** a Pull Request.

### Contribution Areas

  - 🖼️ **Dataset Expansion**: Add more regional and diverse cultural images.
  - 🧠 **Model Improvements**: Experiment with new architectures or training methods.
  - 🌍 **Multilingual Support**: Help MuseForge speak in regional languages.
  - 🧩 **Applications**: Build demos, web apps, or API interfaces.

## 📜 License

This project is licensed under the **MIT License**. This is a permissive license that allows for broad use. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

| Permission          | Status |
| ------------------- | ------ |
| ✅ Commercial Use   | Yes    |
| ✅ Modification     | Yes    |
| ✅ Distribution     | Yes    |
| ✅ Private Use      | Yes    |
| ❌ Liability        | No     |
| ❌ Warranty         | No     |

## 🙏 Acknowledgments

  - A heartfelt thanks to the **Qwen team** for their incredible open-source vision-language model.
  - To the **open-source community** for creating the tools that make projects like this possible.

-----

<div align="center">

**⭐ Star MuseForge if you believe in preserving cultural heritage through AI!**

</div>
# WAVE: Learning Unified & Versatile Audio-Visual Embeddings with Multimodal LLM

<div align="center">

<!-- <img src="assets/logo.png" width="200"/> -->

<div>
    <a href="https://arxiv.org/abs/2509.21990v1" target="_blank">
      <img src="https://img.shields.io/badge/Paper-arXiv-red.svg" alt="Paper arXiv">
    </a>
    <a href="https://huggingface.co/YourUsername/WAVE-7B" target="_blank">
      <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow" alt="Hugging Face Models">
    </a>
    <a href="https://github.com/YourUsername/WAVE" target="_blank">
      <img src="https://img.shields.io/badge/GitHub-Code-blue" alt="GitHub Code">
    </a>
    <img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License">
</div>

</div>

---

## 🔥 News
*   **[2026/02/11]**: 🎉 **WAVE** has been accepted to **ICLR 2026 as an Oral Presentation**!
*   **[2026/02/09]**: 🎉 **WAVE** has been accepted to **ICLR 2026 as an Oral Presentation**!

---

## 📖 Abstract

We introduce **WAVE** (Unified & **V**ersatile **A**udio-**V**isual **E**mbeddings), the first LLM-based embedding model that creates a unified representation space for text, audio, silent video, and synchronized audio-visual modalities.

Unlike previous methods that use separate encoders, WAVE is built upon **Qwen2.5-Omni** and employs a novel **hierarchical feature fusion strategy**. It features a dual-encoder design for audio (Speech + Environmental Sounds) and is trained via a joint multi-modal, multi-task approach.

**Key Capabilities:**
1.  **Any-to-Any Retrieval:** Seamlessly retrieve across text, audio, and video modalities.
2.  **Prompt-Aware Embeddings:** Generate embeddings tailored to user instructions (e.g., specific questions), significantly boosting performance in Multimodal QA.
3.  **SOTA Performance:** WAVE achieves State-of-the-Art results on the **MMEB-v2** video benchmark and excels in audio/video-to-audio retrieval tasks.

<div align="center">
  <!-- 请在仓库的 assets 文件夹中放入论文的 Figure 1 -->
  <img src="assets/architecture.png" width="90%" alt="WAVE Architecture"/>
  <br>
  <em>Overview of WAVE Architecture. It supports text-only, visual-only, audio-only, and audio-visual inputs.</em>
</div>

---

## 🏆 Performance

### Video Embedding Benchmark (MMEB-v2 & LoVR)
WAVE achieves SOTA performance, significantly outperforming industrial-grade models like Seed-1.6-Embedding.

| Model | MMEB-v2 Overall | Retrieval (RET) | QA | LoVR (text-to-clip) |
| :--- | :---: | :---: | :---: | :---: |
| LamRA 7B | 35.0 | 24.3 | 42.6 | 62.9 |
| Seed-1.6-Embedding | 55.3 | 51.3 | 60.9 | - |
| **WAVE 7B (Ours)** | **59.0** | **54.7** | **72.5** | **62.9** |

### Audio & Audio-Visual Retrieval
| Method | AudioCaps (A-RET) | VGGSound (AV-RET) | MMAU (A-QA) |
| :--- | :---: | :---: | :---: |
| WavCaps (Ref) | 42.2 | 10.3 | 71.5 |
| **WAVE 7B (Ours)** | **44.2** | **25.0** | **76.6** |

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/WAVE.git
    cd WAVE
    ```

2.  **Create a virtual environment:**
    ```bash
    conda create -n wave python=3.10
    conda activate wave
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: We recommend installing `flash-attn` for efficient training and inference.*

---

## 🤖 Model Zoo

We provide the pretrained checkpoints on Hugging Face.

| Model | Parameters | Backbone | Download |
| :--- | :---: | :---: | :---: |
| **WAVE-7B** | 7B | Qwen2.5-Omni | [🤗 HuggingFace](https://huggingface.co/YourUsername/WAVE-7B) |

---

## 🚀 Quick Start

WAVE can extract embeddings for text, video, audio, or combined audio-visual inputs. The embeddings are **prompt-aware**, meaning the instruction you provide changes the embedding focus.

```python
import torch
from wave_model import WAVEModel # 假设你的模型定义在 wave_model.py
from transformers import AutoTokenizer

# 1. Load Model
model_path = "YourUsername/WAVE-7B"
model = WAVEModel.from_pretrained(model_path, device_map="cuda", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. Prepare Inputs
# Example: Video + Instruction
video_path = "assets/example_video.mp4"
instruction = "Describe the main event in the video."

# 3. Extract Embeddings
# The model handles tokenization and alignment internally
embedding = model.encode(
    video=video_path, 
    audio=None, 
    text=instruction, 
    modality="video"
)

# 4. Compute Similarity
text_query = "A man is surfing on a big wave."
text_emb = model.encode(text=text_query, modality="text")

similarity = torch.cosine_similarity(embedding, text_emb)
print(f"Similarity: {similarity.item():.4f}")
```

### Prompt-Awareness Example
As shown in our paper (Appendix B), you can guide the embedding:
```python
# Focus on visual objects
emb_visual = model.encode(video=video, text="What animal is in the video?")

# Focus on background sound
emb_audio = model.encode(video=video, text="What sound can be heard in the background?")
```

---

## 📚 Data Preparation

We use a collection of datasets including Panda-70M, MSR-VTT, AudioSet, etc.
Please refer to [DATA.md](docs/DATA.md) for detailed instructions on downloading and preprocessing the training/evaluation data.

---

## 🏋️ Training & Evaluation

### Evaluation
To reproduce the results on MMEB-v2 or other benchmarks:
```bash
# Example: Evaluate on MMEB-v2 Video Track
python evaluate.py --model_path path/to/wave-7b --task mmeb_video --batch_size 16
```

### Training
We support joint multi-modal multi-task training.
```bash
# Multi-node training example
torchrun --nproc_per_node=8 train.py \
    --config configs/train_wave_joint.yaml \
    --output_dir checkpoints/wave_finetuned
```
See [TRAIN.md](docs/TRAIN.md) for hyperparameter details and LoRA configurations.

---

## 📝 Citation

If you find WAVE useful for your research, please consider citing our paper:

```bibtex
@inproceedings{tang2026wave,
  title={WAVE: Learning Unified \& Versatile Audio-Visual Embeddings with Multimodal LLM},
  author={Tang, Changli and Xiao, Qinfan and Mei, Ke and Wang, Tianyi and Rao, Fengyun and Zhang, Chao},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=YourPaperID}
}
```

---

## 🙏 Acknowledgement

This project is built upon:
*   [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5) for the powerful multimodal backbone.
*   [BEATs](https://github.com/microsoft/unilm/tree/master/beats) for the audio event encoder.
*   [MMEB](https://github.com/Alibaba-NLP/MMEB) for the evaluation benchmark.

We thank the authors for their open-source contributions.

---
*License: Apache 2.0*
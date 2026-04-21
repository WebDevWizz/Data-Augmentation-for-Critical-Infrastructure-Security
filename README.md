# Data-Augmentation-for-Critical-Infrastructure-Security

This project explores a **multimodal data augmentation pipeline** for image classification in a critical-infrastructure security scenario.

The original assignment describes a system for improving the detection of suspicious objects and behaviors in power plant surveillance images. As a practical proof of concept, this notebook uses the **Oxford-IIIT Pet** dataset and applies the same workflow to demonstrate how synthetic data can improve classification performance.

## Project Goal

The goal is to compare three training setups:

1. **Baseline**: training on a small real-image sample
2. **Augmented**: training only on synthetic images generated from the sample
3. **Combined**: training on both real and synthetic images

The pipeline is built to show how **captioning, text generation, and text-to-image generation** can be chained together to expand a dataset.

## Pipeline Overview

### 1. Dataset loading
The notebook downloads the **Oxford-IIIT Pet** dataset from `torchvision` and applies basic preprocessing:

- resize to `256 x 256`
- convert to tensor
- normalize images

The full training split contains **3,680 images** and **37 classes**.

### 2. Image captioning with BLIP
A pretrained **BLIP Base** model from Hugging Face is used to generate captions for input images.

To reduce noisy captions, the notebook incorporates the class label into the prompting strategy so that the generated text stays closer to the target category.

### 3. Text variation with GPT-2
A pretrained **GPT-2** model is used to generate textual variants from the original caption.

The prompt is constrained so that the model starts from the class name, improving relevance and reducing unrelated output.

### 4. Synthetic image generation with Stable Diffusion
The generated captions are then passed to **Stable Diffusion v1.5** to create new synthetic images.

These images are used to enlarge the dataset and simulate a richer training distribution.

### 5. Dataset augmentation
A small random sample is created from the original dataset:

- **5 images per class**
- **185 real images total**

For each sampled image, the notebook generates:

- 1 caption
- 2 text variants
- 2 synthetic images per variant

This produces:

- **370 synthetic images**
- **555 images total** when combined with the original sample

### 6. Model training
A pretrained **ResNet18** backbone is used for classification.

To keep the experiment lightweight:

- the backbone is frozen
- only the final fully connected layer is trained

Three models are trained:

- `model_baseline`
- `model_aug`
- `model_total`

### 7. Evaluation
The models are evaluated on the Oxford-IIIT Pet test split using:

- accuracy
- precision
- recall
- weighted F1-score
- confusion matrix

## Results

The notebook reports the following test performance:

| Model | Accuracy | Weighted F1 |
|---|---:|---:|
| Baseline | 0.6860 | 0.6761 |
| Augmented only | 0.5773 | 0.5495 |
| Baseline + Augmented | **0.7656** | **0.7642** |

### Interpretation
The **combined dataset** gives the best results, which suggests that the synthetic samples help the classifier generalize better than the small real-only sample.

The synthetic-only setup performs worse than the baseline, which is expected when training entirely on generated data with limited diversity and potential noise. The best trade-off comes from mixing **real** and **synthetic** examples.

## Main Libraries

- `torch`
- `torchvision`
- `transformers`
- `diffusers`
- `Pillow`
- `matplotlib`
- `scikit-learn`

## Requirements

Install the dependencies with:

```bash
pip install torch torchvision transformers diffusers accelerate pillow matplotlib scikit-learn
```

A GPU is strongly recommended, especially for:

- BLIP caption generation
- Stable Diffusion image synthesis
- model training

## How to Run

1. Clone the repository
2. Install the dependencies
3. Open `PROGETTO.ipynb`
4. Run the notebook cells in order

## Repository Structure

```text
.
├── PROGETTO.ipynb
├── README.md
└── data/
```

## Notes

- The notebook uses a **small random sample** of the original dataset to keep training feasible.
- The project was designed as a **proof of concept** for multimodal augmentation, not as a production-ready security system.
- The strongest result is obtained when **real and synthetic data are combined**.

## Acknowledgements

- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base)
- [GPT-2](https://huggingface.co/gpt2)
- [Stable Diffusion v1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)

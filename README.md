# Stable Diffusion Model Implementation
<div align="center">
    <img src="assets/Stable Diffusion.png">
</div>

This project is an open-source implementation of the Stable Diffusion model, aimed at generating high-quality images from textual descriptions. Our implementation is based on a combination of several key components, including a U-Net architecture, Variational Autoencoder (VAE), CLIP for prompt embedding, a custom scheduler, and time embeddings to facilitate temporal coherence in generated images.



## Acknowledgments

Special thanks to Umar Jamil for his invaluable contributions to this project. We also extend our gratitude to the authors of the original Stable Diffusion paper for their groundbreaking work in the field of text-to-image & image-to-image generation, which has significantly inspired and guided our implementation.



## Project Structure

The project is organized as follows:
```bash
── assets                           # Contains images for README
├── LICENSE
├── models
│   ├── clip                        # CLIP model for prompt embedding
│   ├── stable_diffusion            # Scheduler and pipeline
│   ├── unet                        # U-Net architecture for image generation
│   └── vae                         # VAE decoder and encoder
├── pretrained                      # put pretrained model in this folder
├── README.md
├── requirements.txt
├── sd.ipynb                        # Example usage
├── tokenizer
└── utils                           # Transformer blocks for various utility functions
```

### 1. Environment Setup
We recommend creating a virtual environment and installing the required dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### 2. Training the Model


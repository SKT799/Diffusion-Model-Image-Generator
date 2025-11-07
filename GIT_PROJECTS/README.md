# Diffusion Model Image Generator (MNIST)

This repository contains an implementation of a diffusion-based image generation model trained on the MNIST dataset. The original work was developed as a Colab notebook and exported to a Python script (`diffusion_image_generation_model_on_mnist (1).py`). The model implements a simple conditional DDPM-style U-Net trained to denoise images from Gaussian noise back to MNIST digits.

## Project overview

- Model: U-Net style architecture that predicts the noise component at each diffusion timestep.
- Dataset: MNIST (grayscale 28x28 images).
- Conditioning: Class-conditional generation using one-hot labels, with optional context dropout during training.
- Key features: Sinusoidal time embeddings, down/up blocks with grouped convolutions, and a reversible diffusion sampling loop.

## Files

- `diffusion_image_generation_model_on_mnist (1).py` - Main script exported from Colab. Contains the dataset loading, model definition (UNet and helper blocks), training loop, and sampling functions.

## Requirements

Recommended Python: 3.8+ (tested on 3.8 - 3.11)

Required Python packages (install with pip):

- torch
- torchvision
- einops
- matplotlib
- pillow

You can install them with:

```bash
# Create and activate a venv (Windows cmd.exe)
python -m venv .venv
.\.venv\Scripts\activate

# Install required packages
pip install torch torchvision einops matplotlib pillow
```

Note: Install a CPU or CUDA build of PyTorch as appropriate for your machine. See https://pytorch.org for installation instructions specific to your OS/GPU.

## Quick start

1. Place the repository folder locally (already at `c:\Users\satya\OneDrive\Desktop\GIT_PROJECTS`).
2. (Optional) Create and activate a virtual environment as shown above.
3. Run the training/demo script:

```bash
python "diffusion_image_generation_model_on_mnist (1).py"
```

- The script will download MNIST into a `./data/` folder.
- Training & sampling use CUDA automatically if available.
- The script contains a small training loop (default `epochs = 5`) and periodic sampling/preview.

## How sampling works

- The code implements a forward noising process q(x_t | x_0) and a learned reverse step reverse_q that uses the model's predicted noise to iteratively denoise from random noise to an image.
- There's also a `sample_w` helper that lets you manipulate conditioning strength `w` for class guidance.

## How to commit & push this README to GitHub

If this local folder is already a git repository linked to `https://github.com/SKT799/Diffusion-Model-Image-Generator`, run the following in `cmd.exe` from the project directory:

```cmd
REM Ensure you're in the project folder
cd /d "c:\Users\satya\OneDrive\Desktop\GIT_PROJECTS"

REM Show status, add the new README, commit, and push
git status
git add README.md
git commit -m "Add project README"

git push origin main
```

If your default branch is `master` or another name, replace `main` with the correct branch. If you haven't set up the remote yet, add it with:

```cmd
git remote add origin https://github.com/SKT799/Diffusion-Model-Image-Generator.git
git push -u origin main
```

If push requires authentication, use your GitHub credentials or a PAT (recommended). For more secure workflows, consider setting up an SSH key.

## Notes & tips

- The script is a direct export from Colab and may contain helper imports referring to `utils` modules. Ensure those modules are present in the repo or adjust imports if you split the project into modules.
- For faster experiments, run on a machine with a CUDA-capable GPU and install the corresponding PyTorch CUDA build.
- Consider refactoring into a package layout (e.g., `src/` and a `train.py`, `sample.py`) if you extend the project.

## License

Pick a license for your repo (MIT, Apache-2.0, etc.). This README does not impose a license; add a `LICENSE` file if you want explicit terms.

## Contact / Acknowledgements

- Original notebook exported from Colab.
- If you want, add your contact details or link to the GitHub profile for contributors to reach out.

---

If you'd like, I can also:
- Create a minimal `requirements.txt` for the repo.
- Add a small `.gitignore` tuned for Python projects and notebooks.
- Split the large notebook script into `train.py` / `model.py` / `utils/` for clarity.

Tell me which of the above you'd like me to add next.
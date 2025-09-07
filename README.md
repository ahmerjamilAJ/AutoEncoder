# AutoEncoder


This repo/notebook implements autoencoders using PyTorch in Google Colab:

- **Part 1 — Image Completion (Inpainting):**  
  Train an autoencoder on FFHQ face images (128×128). The input image is corrupted by drawing a small white rectangle; the model learns to reconstruct the original image.

- **Part 2 — Image Denoising (Salt & Pepper):**  
  Train an autoencoder on a person-image subset (COCO). The input image is corrupted with synthetic salt-and-pepper noise; the model learns to output the clean image.

Both parts use a **convolutional autoencoder**:

- **Encoder:** Conv2d blocks with stride=2 to downsample and extract features.  
- **Decoder:** ConvTranspose2d blocks to upsample back to the original resolution.  
- **Activation:** `Sigmoid` at the end to constrain outputs to `[0,1]`.  
- **Loss:** Mean Squared Error (MSE).  
- **Optimizer:** Adam (`lr≈1e-3` recommended).

**Part 1 differences:** input is the masked image; target is the original face.  
**Part 2 differences:** input is a noisy image; target is the clean image.

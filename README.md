# PixelForge
Synthetic Data Generation using GANs for Privacy-Preserving and Data Augmentation (C-NMC Leukemia Dataset)
# PixelForge 🧬
Synthetic Data Generation using GANs for Privacy-Preserving and Data Augmentation  

## 📌 Overview
PixelForge is a deep learning project that leverages Convolutional Neural Networks (CNNs) and Generative Adversarial Networks (GANs) for medical image classification and augmentation.  
The project is tested on the **C-NMC Leukemia Dataset (ALL vs Normal cells)**.

### 🔹 Stage 1
- Built a baseline CNN classifier.
- Achieved **84% accuracy** and **0.90 ROC-AUC**.
- High recall for Leukemia cases (95%) – critical for medical diagnosis.

### 🔹 Stage 2
- Use GANs to generate synthetic cell images.
- Retrain CNN with augmented dataset.
- Evaluate performance improvement.

---

## 🚀 How to Run
### 1. Install dependencies
```bash
pip install -r requirements.txt

📊 Results (Stage 1)

Accuracy: 84%
ROC-AUC: 0.90
Classification Report + Confusion Matrix included in results/.



# # PixelForge-GAN: custom GAN for generating synthetic HEM images
# # ✅ Version for local use (VS Code / Jupyter / Local Python)
# # Place your dataset under: ./PixelForge_Dataset/fold_0/fold_0

# import os, glob, time
# from PIL import Image
# import torch, torch.nn as nn, torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from torchvision.utils import save_image, make_grid
# import random

# # -------------------------
# # Device setup
# # -------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device:", device)

# # -------------------------
# # Utilities & Dataset
# # -------------------------
# class RecursiveImageDataset(Dataset):
#     """Recursively collects image files in `root` (common extensions) and returns transformed tensors."""
#     def __init__(self, root, transform):
#         exts = ["*.bmp", "*.png", "*.jpg", "*.jpeg", "*.BMP", "*.PNG", "*.JPG", "*.JPEG"]
#         self.files = []
#         for e in exts:
#             self.files += glob.glob(os.path.join(root, "**", e), recursive=True)
#         self.files = sorted(self.files)
#         self.transform = transform
#         print(f"[Dataset] Found {len(self.files)} images in {root}")

#     def __len__(self): 
#         return len(self.files)
    
#     def __getitem__(self, idx):
#         img = Image.open(self.files[idx]).convert("RGB")
#         return self.transform(img)

# # -------------------------
# # Normalization helper
# # -------------------------
# class PixelNorm(nn.Module):
#     def forward(self, x, eps=1e-8):
#         return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + eps)

# # -------------------------
# # Generator
# # -------------------------
# class GenBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x): 
#         return self.net(x)

# class PixelForgeGenerator(nn.Module):
#     def __init__(self, z_dim=100, base_ch=64, out_ch=3):
#         super().__init__()
#         self.z_dim = z_dim
#         self.pixelnorm = PixelNorm()
#         self.project = nn.Sequential(
#             nn.ConvTranspose2d(z_dim, base_ch*8, 4, 1, 0),
#             nn.BatchNorm2d(base_ch*8),
#             nn.ReLU(True)
#         )
#         self.up1 = GenBlock(base_ch*8, base_ch*4)
#         self.up2 = GenBlock(base_ch*4, base_ch*2)
#         self.up3 = GenBlock(base_ch*2, base_ch)
#         self.up4 = GenBlock(base_ch, base_ch//2)
#         self.up5 = GenBlock(base_ch//2, base_ch//4)
#         self.to_rgb = nn.Sequential(
#             nn.Conv2d(base_ch//4, out_ch, kernel_size=3, padding=1),
#             nn.Tanh()
#         )

#     def forward(self, z):
#         z = self.pixelnorm(z)
#         x = self.project(z)
#         x = self.up1(x)
#         x = self.up2(x)
#         x = self.up3(x)
#         x = self.up4(x)
#         x = self.up5(x)
#         return self.to_rgb(x)

# # -------------------------
# # Discriminator
# # -------------------------
# def SNConv(in_c, out_c, k=4, s=2, p=1):
#     return nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, k, s, p))

# class PixelForgeDiscriminator(nn.Module):
#     def __init__(self, in_ch=3, base_ch=64):
#         super().__init__()
#         self.net = nn.Sequential(
#             SNConv(in_ch, base_ch), nn.LeakyReLU(0.2, inplace=True),
#             SNConv(base_ch, base_ch*2), nn.LeakyReLU(0.2, inplace=True),
#             SNConv(base_ch*2, base_ch*4), nn.LeakyReLU(0.2, inplace=True),
#             SNConv(base_ch*4, base_ch*8), nn.LeakyReLU(0.2, inplace=True),
#             SNConv(base_ch*8, 1, k=4, s=1, p=0)
#         )

#     def forward(self, x):
#         return self.net(x).view(-1)

# # -------------------------
# # Hinge loss
# # -------------------------
# def d_hinge_loss(real_out, fake_out):
#     loss_real = torch.mean(nn.ReLU()(1.0 - real_out))
#     loss_fake = torch.mean(nn.ReLU()(1.0 + fake_out))
#     return loss_real + loss_fake

# def g_hinge_loss(fake_out):
#     return -torch.mean(fake_out)

# # -------------------------
# # Training function
# # -------------------------
# def train_pixelforge_gan(
#         hem_root="./PixelForge_Dataset/fold_0/fold_0",
#         out_dir="./PixelForge_Synthetic_HEM",
#         z_dim=100,
#         base_ch=64,
#         batch_size=64,
#         epochs=40,
#         lr=2e-4,
#         beta1=0.5,
#         save_every=1,
#         samples_per_epoch=64):

#     os.makedirs(out_dir, exist_ok=True)
#     os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
#     os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ])

#     dataset = RecursiveImageDataset(hem_root, transform)
#     if len(dataset) == 0:
#         raise RuntimeError(f"No images found in {hem_root}. Check your dataset path!")

#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

#     G = PixelForgeGenerator(z_dim=z_dim, base_ch=base_ch).to(device)
#     D = PixelForgeDiscriminator(in_ch=3, base_ch=base_ch).to(device)

#     optG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
#     optD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

#     fixed_z = torch.randn(samples_per_epoch, z_dim, 1, 1, device=device)
#     global_step = 0
#     print(f"[GAN] Training on {len(dataset)} images | epochs={epochs} batch={batch_size}")

#     for epoch in range(1, epochs + 1):
#         G.train(); D.train()
#         d_losses, g_losses = [], []
#         t0 = time.time()

#         for i, real in enumerate(loader):
#             real = real.to(device)
#             b = real.size(0)

#             # --- Discriminator update ---
#             z = torch.randn(b, z_dim, 1, 1, device=device)
#             fake = G(z).detach()
#             real_out = D(real)
#             fake_out = D(fake)
#             lossD = d_hinge_loss(real_out, fake_out)
#             optD.zero_grad(); lossD.backward(); optD.step()

#             # --- Generator update ---
#             z2 = torch.randn(b, z_dim, 1, 1, device=device)
#             fake2 = G(z2)
#             fake_out2 = D(fake2)
#             lossG = g_hinge_loss(fake_out2)
#             optG.zero_grad(); lossG.backward(); optG.step()

#             d_losses.append(lossD.item())
#             g_losses.append(lossG.item())
#             global_step += 1

#             if (i % 200) == 0:
#                 print(f"Epoch {epoch} [{i}/{len(loader)}] D_loss={lossD.item():.4f} G_loss={lossG.item():.4f}")

#         # --- Save samples ---
#         G.eval()
#         with torch.no_grad():
#             fake_fixed = G(fixed_z).cpu()
#         grid = make_grid((fake_fixed + 1) / 2, nrow=8)
#         save_image(grid, os.path.join(out_dir, "samples", f"epoch_{epoch}.png"))

#         # --- Save checkpoints ---
#         if epoch % save_every == 0:
#             torch.save(G.state_dict(), os.path.join(out_dir, "checkpoints", f"G_epoch{epoch}.pth"))
#             torch.save(D.state_dict(), os.path.join(out_dir, "checkpoints", f"D_epoch{epoch}.pth"))

#         print(f"Epoch {epoch}/{epochs} done | D_loss={sum(d_losses)/len(d_losses):.4f} "
#               f"G_loss={sum(g_losses)/len(g_losses):.4f} | time={time.time()-t0:.1f}s")

#     torch.save(G.state_dict(), os.path.join(out_dir, "G_final.pth"))
#     torch.save(D.state_dict(), os.path.join(out_dir, "D_final.pth"))
#     print("✅ Training complete. Models saved to:", out_dir)
#     return G

# # -------------------------
# # Synthetic Image Generation
# # -------------------------
# def generate_synthetic(G, num_images=1000, out_folder="./PixelForge_Synthetic_HEM/gen", batch=16):
#     os.makedirs(out_folder, exist_ok=True)
#     G.eval()
#     z_dim = G.z_dim
#     device = next(G.parameters()).device
#     n_batches = (num_images + batch - 1) // batch
#     idx = 0

#     with torch.no_grad():
#         for _ in range(n_batches):
#             b = min(batch, num_images - idx)
#             z = torch.randn(b, z_dim, 1, 1, device=device)
#             fake = (G(z).cpu() + 1) / 2.0
#             for i in range(b):
#                 save_image(fake[i], os.path.join(out_folder, f"fake_{idx + 1}.png"))
#                 idx += 1

#     print(f"✅ Generated {num_images} images into {out_folder}")
#     return out_folder 

# G = train_pixelforge_gan(
#     hem_root="./archive/fold_0/hem",  # path to your Normal/hem images
#     out_dir="PixelForge_PFGAN",   # where samples & checkpoints will be saved
#     z_dim=100,
#     base_ch=64,
#     batch_size=64,
#     epochs=5,   # start small; increase to 40–100 later
#     lr=2e-4
# )
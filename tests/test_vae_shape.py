from diffusers import AutoencoderKL
import torch

vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse", torch_dtype=torch.bfloat16
)
print("VAE config:")
print("  latent_channels:", vae.config.latent_channels)
print("  in_channels:", vae.config.in_channels)
print("  out_channels:", vae.config.out_channels)

x = torch.randn(1, 3, 512, 512, dtype=torch.bfloat16)
with torch.no_grad():
    lat = vae.encode(x).latent_dist.sample()

print("\nInput shape:", x.shape)
print("Latent shape:", lat.shape)
print("Expected for 512x512: [1, 4, 64, 64]")

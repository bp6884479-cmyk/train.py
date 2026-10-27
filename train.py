import torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

# ====================== CONFIG ======================
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
DATASET_FOLDER = "./dataset"                 # images yahan daalo
OUTPUT_DIR = "./boobs_enhance_lora"
TRIGGER = "boobsenhance"

POSITIVE_PROMPT = f"photo of a woman, {TRIGGER}, large natural breasts, detailed skin, realistic, high quality, 8k"

RANK = 32
LR = 1e-4
EPOCHS = 8
BATCH_SIZE = 1
RESOLUTION = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ====================================================

class SimpleDataset(Dataset):
    def __init__(self, folder):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        self.transform = transforms.Compose([
            transforms.Resize((RESOLUTION, RESOLUTION)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img = self.transform(img)
        return {"pixel_values": img, "prompt": POSITIVE_PROMPT}

def train():
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(DEVICE)

    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=RANK,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none"
    )

    pipe.unet = get_peft_model(pipe.unet, lora_config)
    pipe.unet.print_trainable_parameters()

    dataset = SimpleDataset(DATASET_FOLDER)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=LR)

    pipe.unet.train()

    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(DEVICE, dtype=torch.float16)

            with torch.no_grad():
                latents = pipe.vae.encode(pixel_values).latent_dist.sample() * 0.18215

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=DEVICE).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            text_input = pipe.tokenizer(
                batch["prompt"],
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            encoder_hidden_states = pipe.text_encoder(text_input.input_ids.to(DEVICE))[0]

            noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_postfix(loss=round(loss.item(), 4))

        # har epoch ke baad save
        save_path = f"{OUTPUT_DIR}/epoch_{epoch+1}"
        pipe.unet.save_pretrained(save_path)
        print(f"Saved → {save_path}")

    print("Training Complete!")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train()

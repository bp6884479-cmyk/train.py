from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image

# ====================== SETTINGS ======================
MODEL_ID = "runwayml/stable-diffusion-v1-5"          # base model
LORA_PATH = "./boobs_enhance_lora"                   # apna trained LoRA folder
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

POSITIVE_PROMPT = "photo of a woman, boobsenhance, large natural breasts, detailed skin, realistic, high quality, 8k"
NEGATIVE_PROMPT = "deformed, ugly, bad anatomy, extra limbs, low quality, blurry, oversharpened"

WIDTH = 512
HEIGHT = 768
STEPS = 30
CFG = 7.5
SEED = 42
# ======================================================

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    safety_checker=None
)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(DEVICE)

# LoRA load karo
pipe.load_lora_weights(LORA_PATH)

generator = torch.Generator(DEVICE).manual_seed(SEED)

image = pipe(
    prompt=POSITIVE_PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    width=WIDTH,
    height=HEIGHT,
    num_inference_steps=STEPS,
    guidance_scale=CFG,
    generator=generator
).images[0]

image.save("enhanced_result.png")
print("Image saved as enhanced_result.png")

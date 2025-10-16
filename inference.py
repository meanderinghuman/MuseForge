from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
from PIL import Image
import torch

# Load model and processor
model_path = "./cultural_vlm_working/final_model"
base_model_name = "Qwen/Qwen2-VL-2B-Instruct"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)

print("Loading base model...")
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print("Loading LoRA weights...")
model = PeftModel.from_pretrained(base_model, model_path)
model.eval()

# Load and prepare image
image_path = r"C:\Users\Asus\OneDrive\Desktop\hand-gesture-recognition-mediapipe-main\regional_images\East_India\east_india_001.jpg"
image = Image.open(image_path).convert('RGB')
image = image.resize((336, 336), Image.Resampling.LANCZOS)

# Prepare input
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this cultural image."}
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)

# Generate
print("Generating description...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        pad_token_id=processor.tokenizer.eos_token_id
    )

response = processor.batch_decode(
    outputs[:, inputs.input_ids.shape[1]:],
    skip_special_tokens=True
)[0]

print("\n" + "="*60)
print("Cultural Description:")
print("="*60)
print(response)
print("="*60)
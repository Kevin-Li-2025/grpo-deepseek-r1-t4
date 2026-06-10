import os
import shutil
import torch
from unsloth import FastLanguageModel
from huggingface_hub import hf_hub_download

# Source directory containing the merged weights from previous run
src_model_path = "/kaggle/input/notebooks/kevin250304/grpo-deepseek-r1-t4/grpo_r1_adapter"
dest_model_path = "/kaggle/working/my_model"

print("Copying model weights to writable directory...")
os.makedirs(dest_model_path, exist_ok=True)

# Copy files from read-only mount to writable workspace
for f in os.listdir(src_model_path):
    src_file = os.path.join(src_model_path, f)
    if os.path.isfile(src_file):
        shutil.copy(src_file, dest_model_path)

print("Downloading config.json from unquantized base model repo Qwen/Qwen2.5-Math-1.5B-Instruct...")
# We must use the unquantized base model's config because our saved model.safetensors
# contains merged 16-bit float weights, not 4-bit quantized weights.
hf_hub_download(
    repo_id="Qwen/Qwen2.5-Math-1.5B-Instruct",
    filename="config.json",
    local_dir=dest_model_path
)
hf_hub_download(
    repo_id="Qwen/Qwen2.5-Math-1.5B-Instruct",
    filename="generation_config.json",
    local_dir=dest_model_path
)

print(f"Loading merged model from {dest_model_path} in 16-bit mode...")

# 2. Load the completed model in standard 16-bit mode (fits easily in 16GB T4 VRAM)
max_seq_length = 1024
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = dest_model_path,
    max_seq_length = max_seq_length,
    load_in_4bit = False, # Disable 4bit loading to match FP16 weights
    fast_inference = False
)

# Ensure model is in evaluation mode
model.eval()

# System Prompt instructing the model to think before answering
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# Sample Math Questions for testing
test_questions = [
    "If John has 5 apples and eats 2, then buys 4 more, how many apples does he have?",
    "A bakery sells cakes for $15 each. If they sell 8 cakes and spend $45 on ingredients, what is their net profit?",
    "Solve for x: 3x + 7 = 22. Provide the numeric value."
]

print("\n--- Running Inference Tests ---")
for i, question in enumerate(test_questions, 1):
    print(f"\nQuestion {i}: {question}")
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=512,
            temperature=0.6,
            top_p=0.95,
            use_cache=True
        )
        
    # Decode only the generated part
    generated_tokens = outputs[0][len(inputs[0]):]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print("Response:")
    print(response)
    print("-" * 40)

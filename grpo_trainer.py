import os
import re
import torch
from unsloth import FastLanguageModel

from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset

# 1. Load optimized model (4-bit quantization for T4 GPUs)
max_seq_length = 1024 # Keep sequence length tight for T4 memory
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-Math-1.5B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    fast_inference = False # Disable fast_inference to bypass vLLM dependencies
)

# 2. Add LoRA target modules
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # Memory optimization
)

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

# Format datasets to instruct prompt
def preprocess_dataset(dataset):
    def format_prompt(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]}
            ]
        }
    return dataset.map(format_prompt)

# Load dataset and preprocess
dataset = load_dataset("openai/gsm8k", "main", split="train")
dataset = preprocess_dataset(dataset)

# Helper to extract raw text content from trainer completions
# In modern trl, completions are passed as lists of chat structures (e.g. [[{'role': 'assistant', 'content': '...'}]]
def extract_text_from_completion(content):
    if isinstance(content, list):
        if len(content) > 0 and isinstance(content[0], dict) and "content" in content[0]:
            return content[0]["content"]
        return ""
    elif isinstance(content, str):
        return content
    return str(content)

# Reward 1: Formatting Reward
def format_reward_func(completions, **kwargs):
    rewards = []
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>$"
    for content in completions:
        text = extract_text_from_completion(content)
        # Check if tags exist and are properly structured
        match = re.search(pattern, text, re.DOTALL)
        rewards.append(1.0 if match else 0.0)
    return rewards

# Helper to extract ground truth number from GSM8K answer
def extract_gt(text):
    if "####" in text:
        return text.split("####")[-1].strip()
    return None

# Reward 2: Correctness Reward
def correctness_reward_func(completions, answer, **kwargs):
    rewards = []
    for content, raw_ans in zip(completions, answer):
        text = extract_text_from_completion(content)
        gt = extract_gt(raw_ans)
        # Extract predicted answer inside <answer>...</answer>
        pred_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        pred = pred_match.group(1).strip() if pred_match else None
        
        if gt and pred and gt == pred:
            rewards.append(2.0)
        else:
            rewards.append(0.0)
    return rewards

# Reward 3: Strict Length Penalty (prevents Thought-Length Hacking)
def length_penalty_func(completions, **kwargs):
    rewards = []
    for content in completions:
        text = extract_text_from_completion(content)
        # Penalize if thought reasoning goes way too long
        char_len = len(text)
        if char_len > 1500:
            rewards.append(-1.0)
        elif char_len > 800:
            rewards.append(-0.5)
        else:
            rewards.append(0.0)
    return rewards

# 3. Configure training arguments optimized for T4 GPUs
training_args = GRPOConfig(
    output_dir = "grpo_runs",
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    logging_steps = 1,
    bf16 = False, # Force FP16 on Kaggle T4 GPUs
    fp16 = True,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,
    num_generations = 4, # Small group size to save VRAM
    max_prompt_length = 256,
    max_completion_length = 512,
    max_steps = 100, # Run for 100 steps
    save_strategy = "no",
    use_vllm = False # Bypasses vLLM to ensure compatibility and stability
)

# 4. Initialize and run trainer
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [format_reward_func, correctness_reward_func, length_penalty_func],
    args = training_args,
    train_dataset = dataset,
)

print("Starting GRPO training pipeline on Kaggle...")
trainer.train()

# 5. Save the trained LoRA adapter
print("Saving model adapter...")
model.save_pretrained_merged("grpo_r1_adapter", tokenizer, save_method="lora")
print("GRPO training task complete!")

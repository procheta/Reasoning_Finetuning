# train_qwen2_05b_metamathqa_full_sft.py

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen2-0.5B"
DATASET_NAME = "meta-math/MetaMathQA"
OUTPUT_DIR = "./qwen2-0.5b-metamathqa-full-sft"

# -------------------------
# 1. Load dataset
# -------------------------
dataset = load_dataset(DATASET_NAME, split="train")

# Optional debug subset
# dataset = dataset.select(range(2000))

def format_example(example):
    return {
        "text": f"""### Question:
{example["query"]}

### Solution:
{example["response"]}"""
    }

dataset = dataset.map(format_example, remove_columns=dataset.column_names)

dataset = dataset.train_test_split(test_size=0.02, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# -------------------------
# 2. Tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

# -------------------------
# 3. Load full model
# -------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False

# Optional: useful if using gradient checkpointing
model.gradient_checkpointing_enable()

# -------------------------
# 4. SFT config
# -------------------------
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,

    dataset_text_field="text",
    max_length=1024,
    packing=True,

    num_train_epochs=1,
    per_device_train_batch_size=5,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,

    learning_rate=2e-5,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="adamw_torch",

    logging_steps=20,
    save_steps=500,
    eval_steps=500,
    eval_strategy="steps",
    save_total_limit=2,

    bf16=True,
    fp16=False,

    gradient_checkpointing=True,
    max_grad_norm=1.0,

    report_to="none",
)

# -------------------------
# 5. Trainer
# -------------------------
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

trainer.train()

# -------------------------
# 6. Save full model
# -------------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Full fine-tuned model saved to {OUTPUT_DIR}")

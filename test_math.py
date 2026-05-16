# evaluate_qwen2_metamath_on_gsm8k.py

import re
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "qwen2-0.5b-metamathqa-full-sft/checkpoint-2000"  # change this
DATASET_NAME = "gsm8k"
DATASET_CONFIG = "main"

MAX_SAMPLES = 500   # set None for full test set
MAX_NEW_TOKENS = 512

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

model.eval()

dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test")

if MAX_SAMPLES is not None:
    dataset = dataset.select(range(MAX_SAMPLES))


def build_prompt(question):
    return f"""### Question:
{question}

### Solution:
Let's solve this step by step.
"""


def extract_number(text):
    """
    Extract final numeric answer.
    GSM8K gold answers often contain #### answer.
    """
    if "####" in text:
        text = text.split("####")[-1]

    numbers = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    if len(numbers) == 0:
        return None

    return numbers[-1]


def normalize_number(x):
    if x is None:
        return None
    try:
        return float(x)
    except:
        return None


correct = 0
total = 0

for example in tqdm(dataset):
    question = example["question"]
    gold_answer_text = example["answer"]

    prompt = build_prompt(question)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    generated = output_text[len(prompt):].strip()

    pred = normalize_number(extract_number(generated))
    gold = normalize_number(extract_number(gold_answer_text))

    is_correct = pred == gold

    correct += int(is_correct)
    total += 1

accuracy = correct / total

print("Evaluation dataset:", DATASET_NAME)
print("Total:", total)
print("Correct:", correct)
print("Accuracy:", accuracy)

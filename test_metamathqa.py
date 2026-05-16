# eval_qwen2_05b_metamathqa.py

import re
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "qwen2-0.5b-metamathqa-full-sft/checkpoint-500/"  # your checkpoint path
DATASET_NAME = "meta-math/MetaMathQA"
MAX_SAMPLES = 500  # set None for full eval
MAX_NEW_TOKENS = 512

device = "cuda" if torch.cuda.is_available() else "cpu"

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

dataset = load_dataset(DATASET_NAME, split="train")

# Optional: evaluate on subset
if MAX_SAMPLES is not None:
    dataset = dataset.select(range(MAX_SAMPLES))


def build_prompt(question):
    return f"""### Question:
{question}

### Solution:
"""


def extract_answer(text):
    """
    Extract final numerical/text answer.
    Works for common formats like:
    Final answer: ...
    The answer is ...
    \\boxed{...}
    """
    text = text.strip()

    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed[-1].strip()

    patterns = [
        r"Final answer\s*:\s*(.*)",
        r"The answer is\s*(.*)",
        r"Answer\s*:\s*(.*)",
        r"Therefore,\s*(.*)",
    ]

    for pattern in patterns:
        match = re.findall(pattern, text, flags=re.IGNORECASE)
        if match:
            return match[-1].strip().split("\n")[0]

    # fallback: last non-empty line
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines[-1] if lines else ""


def normalize_answer(ans):
    ans = ans.lower().strip()
    ans = ans.replace(",", "")
    ans = ans.replace("$", "")
    ans = ans.replace("%", "")
    ans = ans.replace(".", "")
    ans = re.sub(r"\s+", " ", ans)
    return ans


correct = 0
total = 0

results = []

for ex in tqdm(dataset):
    question = ex["query"]
    gold_solution = ex["response"]

    prompt = build_prompt(question)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    generated_solution = full_output[len(prompt):].strip()

    pred_answer = extract_answer(generated_solution)
    gold_answer = extract_answer(gold_solution)

    pred_norm = normalize_answer(pred_answer)
    gold_norm = normalize_answer(gold_answer)

    is_correct = pred_norm == gold_norm

    correct += int(is_correct)
    total += 1

    results.append({
        "question": question,
        "gold_solution": gold_solution,
        "generated_solution": generated_solution,
        "gold_answer": gold_answer,
        "pred_answer": pred_answer,
        "correct": is_correct,
    })

accuracy = correct / total if total > 0 else 0

print(f"Total samples: {total}")
print(f"Correct: {correct}")
print(f"Accuracy: {accuracy:.4f}")

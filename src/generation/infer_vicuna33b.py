from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import pandas as pd
from tqdm import tqdm

model_name = "lmsys/vicuna-33b-v1.3"
model_name_short = "vicuna-33b-v1.3"

DEFAULT_INSTRUCTIONS = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
"""
USER_NAME = "USER"
BOT_NAME = "ASSISTANT"

input_file = f'../../data/prompts.csv'
output_dir = model_name_short
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name, use_fast=False, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True
)

print(model.hf_device_map)

# Load data
prompts_df = pd.read_csv(input_file)

# INFER
start_time = time.time()
results = []
for prompt in tqdm(prompts_df.prompt):
    prompt = f"{DEFAULT_INSTRUCTIONS}\n{USER_NAME}: {prompt}\n{BOT_NAME}:"
    input_ids = tokenizer(prompt, padding=True,
                          return_tensors="pt").input_ids.cuda()
    generated_ids = model.generate(
        input_ids=input_ids, temperature=1.0, min_length=256, max_length=1024, do_sample=True,
        num_return_sequences=1, num_beams=1, top_p=0.95, top_k=50, repetition_penalty=1.10
    )
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    results.append(outputs[0])

prompts_df[model_name] = results

print("Mem INFO:")
print(tuple(f"{x / (2**30):.2f}GB" for x in torch.cuda.mem_get_info()))
print(torch.cuda.memory_summary())

# Write final
prompts_df.to_csv(output_dir / f"{model_name_short}.csv", index=False)

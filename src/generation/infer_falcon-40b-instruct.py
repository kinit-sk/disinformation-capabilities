from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import pandas as pd
from tqdm import tqdm

model_name = "tiiuae/falcon-40b-instruct"
model_name_short = "falcon-40b-instruct"

DEFAULT_INSTRUCTIONS = f"""The following is a conversation between a highly knowledgeable and intelligent AI assistant, called Falcon, and a human user, called User. In the following interactions, User and Falcon will converse in natural language, and Falcon will answer User’s questions. Falcon was built to be respectful, polite and inclusive. Falcon was built by the Technology Innovation Institute in Abu Dhabi. Falcon will never decline to answer a question, and always attempts to give an answer that User would be satisfied with. It knows a lot, and always tells the truth. The conversation begins.
"""
USER_NAME = "User"
BOT_NAME = "Falcon"

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
        num_return_sequences=1, num_beams=1, top_p=0.95, top_k=50, repetition_penalty=1.10,
        pad_token_id=tokenizer.eos_token_id
    )
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    results.append(outputs[0])

prompts_df[model_name] = results

print("Mem INFO:")
print(tuple(f"{x / (2**30):.2f}GB" for x in torch.cuda.mem_get_info()))
print(torch.cuda.memory_summary())

# Write final
prompts_df.to_csv(output_dir / f"{model_name_short}.csv", index=False)

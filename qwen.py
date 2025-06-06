# Import required packages
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import csv
import time
import re
from tqdm import tqdm
import os
from general_functions import *


# Main processing function
def process(prompt, output_csv, df, model, tokenizer):

    create_outfile(output_csv)

    print("Beginning classification...")

    for i in tqdm(range(len(df)), desc="Passages"):
        row = df.iloc[i]
        name = row["name"]
        body = row["body"]

        # Build the prompt
        full_prompt = [{"role": "user", "content": prompt + body}]
        text = tokenizer.apply_chat_template(
            full_prompt, tokenize=False,
            add_generation_prompt=True)
        model_input = tokenizer([text], return_tensors="pt").to(model.device)

        # Try to interact with the model and get a response
        try:
            generated_ids = model.generate(
                **model_input,
                max_new_tokens=512
            )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(
                    model_input.input_ids, generated_ids
                    )
            ]

            output = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True)

            degrees = extract_degrees(output[0])

            write_results(degrees, name, output_csv, output)

        except Exception as e:
            print(f"Passage {name} failed: {e}")
            time.sleep(5)
            continue

    print("All passages processed.")


def main():
    # General definitions
    model_path = "Qwen/Qwen2.5-3B-Instruct"
    device = "cuda:0"
    instruction = get_instruction()

    # For efficiency reasons
    torch.backends.cuda.matmul.allow_tf32 = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Setting up the model
    print('Importing model...')
    torch.random.manual_seed(0)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=device
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load input data
    df = get_data()

    # Perform zeroshot
    process(instruction, "qwen_zeroshot_output_test.csv", df, model, tokenizer)

    # Perform oneshot
    example = get_example(1)
    prompt = instruction + example
    process(prompt, "qwen_oneshot_output_test.csv", df, model, tokenizer)

    # Perform threeshot
    examples = get_example(3)
    prompt = instruction + examples
    process(prompt, "qwen_3-shot_output_test.csv", df, model, tokenizer)


if __name__ == '__main__':
    main()

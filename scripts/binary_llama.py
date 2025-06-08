# Import required packages
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import csv
import time
import re
from tqdm import tqdm
from huggingface_hub import login
import os
from binary_general_functions import *


# Main processing function
def process(prompt, output_csv, df, pipe, generation_args):

    create_outfile(output_csv)

    print("Beginning classification...")

    for i in tqdm(range(len(df)), desc="Passages"):
        row = df.iloc[i]
        name = row["name"]
        body = row["body"]

        # Build the prompt
        full_prompt = [{"role": "user", "content": prompt + body}]

        # Try to interact with the model and get a response
        try:
            output = pipe(full_prompt, **generation_args)

            degrees = extract_degrees(output[0]['generated_text'])

            write_results(degrees, name, output_csv, output)

        except Exception as e:
            print(f"Passage {name} failed: {e}")
            time.sleep(5)
            continue

    print("All passages processed.")


def main():
    # Get access to the Llama model
    with open('llama_token.txt', 'r', encoding='utf-8') as file:
        token_string = file.read()
    login(token=token_string)

    # General definitions
    model_path = "meta-llama/Llama-3.2-3B-Instruct"
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
        device_map=device,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    # Load input data
    df = get_data()

    # Perform zeroshot
    process(
        instruction, "llama_bi_zeroshot_output_test.csv", df, pipe,
        generation_args
    )

    # Perform oneshot
    example = get_example(1)
    prompt = instruction + example
    process(
        prompt, "llama_bi_oneshot_output_test.csv", df, pipe, generation_args
    )

    # Perform threeshot
    examples = get_example(3)
    prompt = instruction + examples
    process(
        prompt, "llama_bi_3-shot_output_test.csv", df, pipe, generation_args
    )


if __name__ == '__main__':
    main()

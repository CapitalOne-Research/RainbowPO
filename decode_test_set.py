# from vllm import LLM, SamplingParams
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import torch
transformers.logging.set_verbosity_error()

from datasets import load_dataset, load_from_disk
import argparse
import json

parser = argparse.ArgumentParser(description='Decode with vllm')
parser.add_argument('--data_dir', type=str, default="<some path>/trl/examples/datasets/UltraFeedback_armorm_trl",
                    help='Directory containing the data')
parser.add_argument('--model_path', type=str, default="google/gemma-2-9b-it",
                    help='Path to the LLM model')
parser.add_argument('--model_output_name', type=str, default="Llama3-Instruct",
                    help='Model nickname in the output directory')
parser.add_argument('--temperature', type=float, default=0.7,
                    help='Temperature for sampling')
parser.add_argument('--top_p', type=float, default=0.95,
                    help='Top-p probability for sampling')
parser.add_argument('--max_tokens', type=int, default=2048,
                    help='Maximum number of tokens to generate')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed')
parser.add_argument('--output_dir', type=str, default="<some path>/evaluations/reward_mean/",
                    help='output_dir')
args = parser.parse_args()

print(args)

data_dir = args.data_dir

tokenizer = AutoTokenizer.from_pretrained(args.model_path,device_map="auto")

model = AutoModelForCausalLM.from_pretrained(args.model_path,device_map="auto")

# llm = LLM(model=args.model_path, gpu_memory_utilization=0.95, tensor_parallel_size=8)

# tokenizer = llm.get_tokenizer()

test_dataset= load_from_disk(data_dir, split='test')

prompts = list(set(test_dataset['prompt']))

conversations = [tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True) for prompt in prompts]

# sampling_params = SamplingParams(temperature=args.temperature, 
#                                  top_p=args.top_p, 
#                                  max_tokens=args.max_tokens, 
#                                  seed=args.seed,)
# outputs = llm.generate(conversations, sampling_params)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer_sft,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="auto",
)

for input in conversations:
    outputs = pipe(
            input,
            max_new_tokens=args.max_token,
            eos_token_id=terminators,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
        )

import pdb;pdb.set_trace()

# Save the outputs as a JSON file.
output_data = []
for i, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    output_data.append({
        'prompt': prompts[i],
        "format_prompt": prompt,
        'generated_text': generated_text,
    })

output_file = args.model_output_name+f'_output_{args.seed}.json'

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

with open(os.path.join(args.output_dir, output_file), 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"Outputs saved to {os.path.join(args.output_dir, output_file)}")

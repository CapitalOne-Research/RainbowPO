# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import multiprocessing
from dataclasses import dataclass, field

from datasets import load_dataset,load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import ModelConfig, ORPOConfig, ORPOTrainer, get_peft_config

from accelerate import PartialState


@dataclass
class ScriptArguments:
    dataset: str = field(
        default="<some path>/trl/examples/datasets/UltraFeedback_armorm_trl",
        metadata={"help": "The name of the dataset to use."},
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, ORPOConfig, ModelConfig))
    args, orpo_args, model_config = parser.parse_args_into_dataclasses()

    orpo_args.lr_scheduler_type = "cosine"

    ################
    # Model & Tokenizer
    ################
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path)
    peft_config = get_peft_config(model_config)
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"


    ################
    # Dataset
    ################
    ds = load_from_disk(args.dataset)

    if orpo_args.debug:
        for key in ds:
            ds[key] = ds[key].select(range(50))
    if tokenizer.chat_template is None:
        #tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        raise ValueError("Tokenizer chat template is not set.")
    # def process(row):
    #     row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
    #     row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
    #     return row
    def process(row):
        row["prompt"] = tokenizer.apply_chat_template(row["chosen"][:-1], tokenize=False)
        row["chosen"] = tokenizer.apply_chat_template([row["chosen"][-1]], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template([row["rejected"][-1]], tokenize=False)

        # Add these lines to reduce bos_tokens.
        if row["chosen"].startswith(tokenizer.bos_token):
            row["chosen"] = row["chosen"][len(tokenizer.bos_token):]
        if row["rejected"].startswith(tokenizer.bos_token):
            row["rejected"] = row["rejected"][len(tokenizer.bos_token):]
        return row

    ds = ds.map(
        process,
        num_proc=16,
        load_from_cache_file=False,
    )
    train_dataset = ds["train"]
    eval_dataset = ds["test"]

    ################
    # Training
    ################
    trainer = ORPOTrainer(
        model,
        args=orpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    # train and save the model
    trainer.train()
    trainer.save_model(orpo_args.output_dir)

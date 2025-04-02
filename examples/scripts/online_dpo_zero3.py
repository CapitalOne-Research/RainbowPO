from dataclasses import dataclass
from typing import Optional

from datasets import load_dataset,load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from trl import ModelConfig
from trl.commands.cli_utils import TrlParser
from trl.trainer import OnlineDPOConfig, OnlineDPOTrainer
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE

from accelerate import PartialState


"""
python examples/scripts/online_dpo.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --learning_rate 3e-6 \
    --output_dir models/minimal/online_dpo \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --total_episodes 30000 \
    --model_name_or_path EleutherAI/pythia-14m \
    --reward_model_path EleutherAI/pythia-14m \
    --non_eos_penalty \
    --stop_token eos \
    --response_length 53 \
    --sanity_check
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/online_dpo.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --learning_rate 3e-6 \
    --output_dir models/minimal/online_dpo \
    --per_device_train_batch_size 16 \
    --local_rollout_forward_batch_size 32 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --gradient_accumulation_steps 4 \
    --total_episodes 1000000 \
    --model_name_or_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr  \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --save_strategy no \
    --non_eos_penalty \
    --stop_token eos \
    --beta 0.1 \
    --response_length 53 \
    --push_to_hub
"""


@dataclass
class ScriptArguments:
    dataset_name: str = "<some path>/princeton-nlp_llama3-ultrafeedback-armorm"
    dataset_text_field: str = "prompt"
    dataset_train_split: str = "train"
    dataset_test_split: Optional[str] = None
    max_length: int = 1024


def prepare_dataset(dataset, tokenizer, dataset_text_field, num_proc):
    """pre-tokenize the dataset before training; only collate during training"""

    def tokenize(element):
        outputs = tokenizer(
            element[dataset_text_field],
            padding=False,
        )
        return {"input_ids": outputs["input_ids"]}

    return dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=False,
    )


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, OnlineDPOConfig, ModelConfig))
    args, config, model_config = parser.parse_args_and_config()

    config.lr_scheduler_type = "cosine"

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

    reward_model = AutoModelForSequenceClassification.from_pretrained(config.reward_model_path, num_labels=1)
    ref_model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path)
    ref_model.eval()
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path)

    distributed_state = PartialState()
    model.to(distributed_state.device)
    reward_model.to(distributed_state.device)

    ################
    # Dataset
    ################
    raw_datasets = load_dataset(args.dataset_name)
    #raw_datasets = load_from_disk(args.dataset_name)

    if config.sanity_check:
        for key in raw_datasets:
            raw_datasets[key] = raw_datasets[key].select(range(1024))
    
    train_dataset = raw_datasets[args.dataset_train_split]
    train_dataset = prepare_dataset(train_dataset, tokenizer, args.dataset_text_field, config.dataset_num_proc)

    if args.dataset_test_split is not None:
        eval_dataset = raw_datasets[args.dataset_test_split]
        eval_dataset = prepare_dataset(eval_dataset, tokenizer, args.dataset_text_field, config.dataset_num_proc)
    else:
        eval_dataset = None

    print("Finish loading models and datasets, start training now.\n")
    ################
    # Training
    ################

    trainer = OnlineDPOTrainer(
        config=config,
        tokenizer=tokenizer,
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()

    if not config.sanity_check:
        trainer.save_model(config.output_dir)
        if config.push_to_hub:
            trainer.push_to_hub()
        trainer.generate_completions()

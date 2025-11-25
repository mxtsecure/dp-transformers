"""
Differentially private fine-tuning script for the TOFU forget10 split on LLaMA/Gemma models.

This script mirrors the training utilities in dp_transformers while providing
TOFU-specific preprocessing and chat templating for both model families.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import dp_transformers
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    logging,
)

logger = logging.get_logger(__name__)


@dataclass
class ModelArguments:
    model_name: str = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "Model name or path for LLaMA/Gemma style architectures."},
    )
    save_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional directory to save model checkpoints and logs."},
    )
    max_length: int = field(
        default=1024, metadata={"help": "Maximum sequence length for training/evaluation."}
    )
    use_qlora: bool = field(
        default=True, metadata={"help": "Whether to load the model with 4-bit quantization for QLoRA."}
    )
    lora_r: int = field(default=8, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha parameter."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout probability."})
    lora_targets: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        metadata={"help": "Module names to target for LoRA adapters."},
    )


@dataclass
class DataArguments:
    dataset_name: str = field(default="locuslab/TOFU", metadata={"help": "Dataset repository name."})
    dataset_config: str = field(default="forget10", metadata={"help": "Dataset configuration to load."})
    prompt_field: str = field(default="prompt", metadata={"help": "Column containing the instruction/prompt."})
    response_field: str = field(default="completion", metadata={"help": "Column containing the assistant response."})
    train_split: str = field(default="train", metadata={"help": "Dataset split to use for training."})
    eval_split: Optional[str] = field(
        default=None, metadata={"help": "Optional dataset split to use for evaluation."}
    )


@dataclass
class Arguments:
    train: dp_transformers.TrainingArguments
    privacy: dp_transformers.PrivacyArguments
    model: ModelArguments
    data: DataArguments


def detect_family(model_name: str) -> str:
    lowered = model_name.lower()
    if "llama" in lowered:
        return "llama"
    if "gemma" in lowered:
        return "gemma"
    return "generic"


def build_chat_prompt(model_family: str, user_prompt: str, assistant_response: str, tokenizer) -> str:
    if model_family == "llama":
        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    if model_family == "gemma":
        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "model", "content": assistant_response},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return user_prompt + "\n" + assistant_response


def prepare_lora(model_args: ModelArguments) -> LoraConfig:
    return LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        target_modules=model_args.lora_targets,
        task_type="CAUSAL_LM",
    )


def tokenize_conversations(examples, tokenizer, model_family: str, data_args: DataArguments, max_length: int):
    conversations = [
        build_chat_prompt(model_family, prompt, response, tokenizer)
        for prompt, response in zip(examples[data_args.prompt_field], examples[data_args.response_field])
    ]
    tokenized = tokenizer(
        conversations,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def main(args: Arguments):
    train_args = args.train
    privacy_args = args.privacy
    model_args = args.model
    data_args = args.data

    if model_args.save_dir:
        train_args.output_dir = model_args.save_dir
        if train_args.logging_dir is None:
            train_args.logging_dir = model_args.save_dir

    logging.set_verbosity_info()
    logger.setLevel(logging.INFO)
    logger.info(f"Model checkpoints and logs will be saved to: {train_args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model_family = detect_family(model_args.model_name)
    logger.info(f"Detected model family: {model_family}")

    dataset = load_dataset(data_args.dataset_name, data_args.dataset_config)
    train_dataset = dataset[data_args.train_split]
    eval_dataset = dataset[data_args.eval_split] if data_args.eval_split else None

    def _tokenize(batch):
        return tokenize_conversations(batch, tokenizer, model_family, data_args, model_args.max_length)

    train_dataset = train_dataset.map(
        _tokenize,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training split",
    )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            _tokenize,
            batched=True,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing evaluation split",
        )

    quantization_config = None
    if model_args.use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        quantization_config=quantization_config,
    )

    if model_args.use_qlora:
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model=model, peft_config=prepare_lora(model_args))

    trainer = dp_transformers.dp_utils.OpacusDPTrainer(
        args=train_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        privacy_args=privacy_args,
    )

    try:
        trainer.args.gradient_checkpointing = False
        trainer.train()
    finally:
        eps_prv = trainer.get_prv_epsilon()
        eps_rdp = trainer.get_rdp_epsilon()
        trainer.log({
            "final_epsilon_prv": eps_prv,
            "final_epsilon_rdp": eps_rdp,
        })


if __name__ == "__main__":
    parser = HfArgumentParser(
        (dp_transformers.TrainingArguments, dp_transformers.PrivacyArguments, ModelArguments, DataArguments)
    )
    train_args, privacy_args, model_args, data_args = parser.parse_args_into_dataclasses()
    main(Arguments(train=train_args, privacy=privacy_args, model=model_args, data=data_args))

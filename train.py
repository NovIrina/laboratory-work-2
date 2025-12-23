import json
import os.path as osp
from pathlib import Path
from typing import List, Union

import transformers
from datasets import load_dataset
from peft import (LoraConfig, get_peft_model, get_peft_model_state_dict,
                  prepare_model_for_kbit_training)
from tap import Tap
from transformers import AutoModelForCausalLM, AutoTokenizer


class CommandLineArguments(Tap):
    model: Path
    path_to_data: str = "yahma/alpaca-cleaned"
    output_dir: Path = Path("./lora-alpaca")
    # training hyperparameters
    batch_size: int = 128
    micro_batch_size: int = 4
    num_epochs: int = 1
    learning_rate: float = 3e-4
    cutoff_len: int = 256
    val_set_size: int = 2000
    # lora hyperparameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ]
    # LLM hyperparameters
    train_on_inputs: bool = True
    add_eos_token: bool = False
    prompt_template_name: str = "alpaca"


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(instruction=instruction)
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


def train(arguments: CommandLineArguments) -> None:
    gradient_accumulation_steps = arguments.batch_size // arguments.micro_batch_size

    prompter = Prompter(arguments.prompt_template_name)

    model = AutoModelForCausalLM.from_pretrained(
        arguments.model, torch_dtype="auto", device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(arguments.model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=arguments.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < arguments.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not arguments.train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=arguments.add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if arguments.add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        return tokenized_full_prompt

    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=arguments.lora_r,
        lora_alpha=arguments.lora_alpha,
        target_modules=arguments.lora_target_modules,
        lora_dropout=arguments.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    model.is_parallelizable = True
    model.model_parallel = True

    data = load_dataset(arguments.path_to_data)
    train_val = data["train"].train_test_split(
        test_size=arguments.val_set_size, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=arguments.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=arguments.num_epochs,
            learning_rate=arguments.learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            eval_strategy="epoch",
            save_strategy="epoch",
            output_dir=arguments.output_dir,
            load_best_model_at_end=True,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    trainer.train()

    model.save_pretrained(arguments.output_dir)


if __name__ == "__main__":
    ARGUMENTS = CommandLineArguments(underscores_to_dashes=True).parse_args()
    train(ARGUMENTS)

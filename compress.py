from pathlib import Path
from tap import Tap

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer


class CommandLineArguments(Tap):
    path_to_model: Path
    path_to_save: Path


def compress_model(path_to_model: Path) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        path_to_model,
        torch_dtype="auto",
        quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(path_to_model)
    return model, tokenizer


def main(arguments: CommandLineArguments):
    model, tokenizer = compress_model(arguments.path_to_model)
    model.save_pretrained(arguments.path_to_save)
    tokenizer.save_pretrained(arguments.path_to_save)


if __name__ == "__main__":
    ARGUMENTS = CommandLineArguments(underscores_to_dashes=True).parse_args()
    main(ARGUMENTS)

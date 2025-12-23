from pathlib import Path
from typing import Optional

import lm_eval
from lm_eval.api.registry import get_model
from lm_eval.models.huggingface import HFLM
from tap import Tap


class CommandLineArguments(Tap):
    path_to_original_model: Path
    path_to_compressed_model: Path
    peft_model: Optional[Path] = None
    batch_size: int


def load_models(
    path_to_original_model: Path,
    path_to_compressed_model: Path,
    batch_size: int,
    peft_model: Optional[Path] = None,
) -> tuple[HFLM, HFLM]:
    original_model = get_model("hf").create_from_arg_string(
        f"pretrained={path_to_original_model},device=cuda,parallelize=True,trust_remote_code=True",
        additional_config={"batch_size": batch_size, "max_batch_size": batch_size},
    )

    if peft_model:
        compressed_model = get_model("hf").create_from_arg_string(
            f"pretrained={path_to_compressed_model},peft={peft_model},device=cuda,parallelize=True,trust_remote_code=True",
            additional_config={"batch_size": batch_size, "max_batch_size": batch_size},
        )
    else:
        compressed_model = get_model("hf").create_from_arg_string(
            f"pretrained={path_to_compressed_model},device=cuda,parallelize=True,trust_remote_code=True",
            additional_config={"batch_size": batch_size, "max_batch_size": batch_size},
        )

    return original_model, compressed_model


def check_memory_footprint(original_model: HFLM, compressed_model: HFLM) -> None:
    original_model_memory_footprint = original_model.model.get_memory_footprint()
    compressed_model_memory_footprint = compressed_model.model.get_memory_footprint()

    compression_ratio = (
        original_model_memory_footprint / compressed_model_memory_footprint
    )
    print(f"Compression Ratio: {compression_ratio}")
    return compression_ratio


def check_quality(
    path_to_original_model: Path,
    path_to_compressed_model: Path,
    original_model: HFLM,
    compressed_model: HFLM,
    batch_size: int,
    peft_model: Optional[Path] = None,
) -> None:
    original_results = lm_eval.simple_evaluate(
        model=original_model,
        model_args=f"pretrained={path_to_original_model}",
        tasks="mmlu",
        batch_size=batch_size,
        device="cuda",
        num_fewshot=0,
    )["results"]["mmlu"]["acc,none"]

    if peft_model:
        compressed_results = lm_eval.simple_evaluate(
            model=compressed_model,
            model_args=f"pretrained={path_to_compressed_model},peft={peft_model}",
            tasks="mmlu",
            batch_size=batch_size,
            device="cuda",
            num_fewshot=0,
        )["results"]["mmlu"]["acc,none"]
    else:
        compressed_results = lm_eval.simple_evaluate(
            model=compressed_model,
            model_args=f"pretrained={path_to_compressed_model}",
            tasks="mmlu",
            batch_size=batch_size,
            device="cuda",
            num_fewshot=0,
        )["results"]["mmlu"]["acc,none"]

    performance_drop = (original_results - compressed_results) / original_results
    return performance_drop


def get_score(compression_ratio: float, performance_drop: float) -> float:
    score = compression_ratio / (1 + performance_drop)
    return score


def print_report(
    compression_ratio: float, performance_drop: float, score: float
) -> None:
    print(f"Compression Ratio: {compression_ratio}")
    print(f"Performance Drop: {performance_drop}")
    print(f"Score: {score}")


def main(arguments: CommandLineArguments):
    original_model, compressed_model = load_models(
        arguments.path_to_original_model,
        arguments.path_to_compressed_model,
        arguments.batch_size,
        arguments.peft_model,
    )
    compression_ratio = check_memory_footprint(original_model, compressed_model)
    performnace_drop = check_quality(
        arguments.path_to_original_model,
        arguments.path_to_compressed_model,
        original_model,
        compressed_model,
        arguments.batch_size,
    )
    score = get_score(compression_ratio, performnace_drop)
    print_report(compression_ratio, performnace_drop, score)


if __name__ == "__main__":
    ARGUMENTS = CommandLineArguments(underscores_to_dashes=True).parse_args()
    main(ARGUMENTS)

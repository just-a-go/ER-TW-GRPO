# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import math
from functools import partial

import torch
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration
from transformers import AutoConfig

from math_verify import parse, verify
from open_r1.trainer.grpo_config import GRPOConfig
from open_r1.trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

class ProcessLogger:
    def __init__(self, prefix=""):
        self.pid = os.getpid()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(
            os.getenv("PRIVATE_DATA_ROOT"),
            os.getenv("WANDB_NAME"),
            "debug_logs"
        )
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{prefix}_{timestamp}_pid{self.pid}.log")
        
    def log(self, message):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

accuracy_logger = ProcessLogger("accuracy")
format_logger = ProcessLogger("format")
description_logger = ProcessLogger("description")

@dataclass
class GRPOScriptArguments(ScriptArguments):

    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    jsonl_path: Optional[str] = field(
        default=None,
        metadata={"help": "json file path"},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False
    loss_type: str = "grpo"
    alpha: float = 1.4
    generate_temperature: float = 1.0
    question_type: str = "mixed"
    use_epsilon: bool = False
    use_dynamic_sampling: bool = False
    train_samples: int = 2000
    cf_local_data: Optional[str] = None
    cf_lambda: float = 0.0
    cf_local_batch_size: int = 1

def accuracy_reward(completions, solution, **kwargs):
    """
    Reward function that checks if the completion is correct, supporting partial credit for subset matches.
    Returns a score between 0 and 1, where:
    - 1.0: perfect match
    - 0.0-1.0: partial match (when model answer is a correct subset)
    - 0.0: incorrect match
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        verification_method = "none"
        
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
                verification_method = "symbolic"
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching with partial credit
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r"<answer>(.*?)</answer>", sol, re.DOTALL)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                # Extract answer from content if it has think/answer tags
                content_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
                student_answer = content_match.group(1).strip() if content_match else content.strip()

                # Convert answers to sets for comparison
                ground_truth_set = set(option.strip() for option in ground_truth.replace(' ', '').split(','))
                student_answer_set = set(option.strip() for option in student_answer.replace(' ', '').split(','))

                # Check if student answer is a subset of correct answers
                if student_answer_set.issubset(ground_truth_set):
                    # Calculate partial credit: number of correct answers / total number of correct answers
                    reward = len(student_answer_set) / len(ground_truth_set)
                    verification_method = "string_matching"
                else:
                    reward = 0.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail

        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            if os.getenv("DEBUG_MODE") == "true":
                accuracy_logger.log(f"video path: {kwargs.get('video_path', 'N/A')}")
                accuracy_logger.log(f"Model output: {content}")
                accuracy_logger.log(f"Solution: {sol}")
                accuracy_logger.log(f"Calculated reward: {reward}")
                accuracy_logger.log(f"Verification method: {verification_method}")
                if verification_method == "string_matching":
                    accuracy_logger.log(f"Ground truth set: {ground_truth_set}")
                    accuracy_logger.log(f"Student answer set: {student_answer_set}")
    
    return rewards

def origin_accuracy_reward(completions, solution, **kwargs):
    """
    Reward function that checks if the completion is correct, supporting partial credit for subset matches.
    Returns a score between 0 and 1, where:
    - 1.0: perfect match
    - 0.0-1.0: partial match (when model answer is a correct subset)
    - 0.0: incorrect match
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        verification_method = "none"
        
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
                verification_method = "symbolic"
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching with partial credit
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r"<answer>(.*?)</answer>", sol, re.DOTALL)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                # Extract answer from content if it has think/answer tags
                content_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
                student_answer = content_match.group(1).strip() if content_match else content.strip()

                # Convert answers to sets for comparison
                ground_truth_set = set(option.strip() for option in ground_truth.replace(' ', '').split(','))
                student_answer_set = set(option.strip() for option in student_answer.replace(' ', '').split(','))

                # Check if student answer is a subset of correct answers
                if student_answer_set.issubset(ground_truth_set):
                    # Calculate partial credit: number of correct answers / total number of correct answers
                    reward = len(student_answer_set) / len(ground_truth_set)
                    if reward < 1:
                        reward = 0
                    verification_method = "string_matching"
                else:
                    reward = 0.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail

        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            if os.getenv("DEBUG_MODE") == "true":
                accuracy_logger.log(f"video path: {kwargs.get('video_path', 'N/A')}")
                accuracy_logger.log(f"Model output: {content}")
                accuracy_logger.log(f"Solution: {sol}")
                accuracy_logger.log(f"Calculated reward: {reward}")
                accuracy_logger.log(f"Verification method: {verification_method}")
                if verification_method == "string_matching":
                    accuracy_logger.log(f"Ground truth set: {ground_truth_set}")
                    accuracy_logger.log(f"Student answer set: {student_answer_set}")
    
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>[\s!]*$"
    completion_contents = [completion[0]["content"] for completion in completions]
    
    def check_format(content):
        # Check for duplicate tags
        if content.count("<think>") > 1 or content.count("<answer>") > 1:
            return False
        # Use more lenient pattern matching
        return bool(re.search(pattern, content.strip(), re.DOTALL))
    
    rewards = [1.0 if check_format(content) else 0.0 for content in completion_contents]
    
    if os.getenv("DEBUG_MODE") == "true":
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        for content, reward in zip(completion_contents, rewards):
            log_entry = (
                f"{current_time} | "
                f"Model output: {content} | "
                f"Calculated reward: {reward} | "
                f"Format match: {'Yes' if reward == 1.0 else 'No'}"
            )
            if reward == 0.0:
                think_count = content.count("<think>")
                answer_count = content.count("<answer>")
                log_entry += (
                    f" | Number of <think> tags: {think_count} | "
                    f"Number of <answer> tags: {answer_count}"
                )
            format_logger.log(log_entry)
    return rewards


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "origin_accuracy": origin_accuracy_reward,
    "format": format_reward,
}

from datasets import Dataset, DatasetDict
import json

def create_dataset_from_jsonl_simple(jsonl_path, question_type, train_samples=2000):
    base_dataset = Dataset.from_json(jsonl_path)
    
    # No filtering when question_type="mixed"
    if question_type != "mixed":
        def is_target_choice_type(example):
          # Extract answer from solution tag
          answer = example['solution'].replace('<answer>', '').replace('</answer>', '')
          # Check if answer contains comma to determine if it's multiple choice
          is_multiple = ',' in answer
          # question_type="single" for single choice, "multiple" for multiple choice
          return not is_multiple if question_type == "single" else is_multiple

        base_dataset = base_dataset.filter(is_target_choice_type)
    
    if len(base_dataset) > train_samples:
        base_dataset = base_dataset.shuffle(seed=42).select(range(train_samples))

    return DatasetDict({
      "train": base_dataset
    })

def main(script_args, training_args, model_args):    
    
    # Add debug log settings
    if os.getenv("DEBUG_MODE") == "true":
        log_dir = os.path.join(training_args.output_dir, "debug_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        os.environ["LOG_PATH"] = log_path
        print(f"Debug logs will be saved to: {log_path}")

    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    
    if not hasattr(model_args, 'train_samples'):
        model_args.train_samples = 2000

    if not hasattr(model_args, 'question_type'):
        model_args.question_type = "mixed"

    if script_args.jsonl_path:
        # # load dataset from jsonl
        print(model_args.question_type)
        dataset = create_dataset_from_jsonl_simple(script_args.jsonl_path, model_args.question_type, model_args.train_samples)
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    if not hasattr(model_args, 'alpha'):
        model_args.alpha = 1.4
    
    if not hasattr(model_args, 'generate_temperature'):
        model_args.generate_temperature = 1.0

    if model_args.question_type == "single":
        QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (select one letter) in <answer> </answer> tags."
        print("single")
    elif model_args.question_type == "multiple":
        QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (select letters separated by ,) in <answer> </answer> tags."
        print("multiple")
    else:  # mixed
        QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (letters separated by , if multiple) in <answer> </answer> tags."

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }
    
    def make_conversation_video(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        # {"type": "video", "video": example["video"]},
                        # {"type": "video", "bytes": open(example["video"],"rb").read()},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
    }
    
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "user", "content": example["problem"]},
            ],
        }

    if "image" in dataset[script_args.dataset_train_split].features:
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
    elif "video" in dataset[script_args.dataset_train_split].features:
        dataset = dataset.map(
            make_conversation_video,
        )
    else:
        dataset = dataset.map(make_conversation)
        # dataset = dataset.remove_columns("messages")
    
    # import pdb; pdb.set_trace()

    trainer_cls = Qwen2VLGRPOTrainer
    cf_kwargs = {}
    if not math.isfinite(model_args.cf_lambda) or model_args.cf_lambda < 0:
        raise ValueError("cf_lambda must be finite and nonnegative")
    if model_args.cf_lambda > 0:
        from open_r1.trainer.cf_trainer import CFQwen2VLGRPOTrainer
        trainer_cls = CFQwen2VLGRPOTrainer
        cf_kwargs = dict(
            cf_local_data=model_args.cf_local_data,
            cf_lambda=model_args.cf_lambda,
            cf_local_batch_size=model_args.cf_local_batch_size,
        )

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        loss_type=model_args.loss_type,
        alpha=model_args.alpha,
        generate_temperature=model_args.generate_temperature,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        **cf_kwargs,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    main(script_args, training_args, model_args)

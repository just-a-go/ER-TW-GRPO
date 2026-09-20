import os
import torch
import json
from tqdm import tqdm
import re
import random
import math
from datetime import datetime
from transformers import (
    Qwen2VLForConditionalGeneration, 
    Qwen2_5_VLForConditionalGeneration, 
    AutoTokenizer, 
    AutoProcessor
)
from qwen_vl_utils import process_vision_info
from typing import Optional
import logging
import logging
import argparse
import sys
import glob

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='evaluation.log',  # Specify log file name
    filemode='w'  # 'w' for write mode, overwrites log file on each run
)

os.environ['VIDEO_MAX_PIXELS'] = str(256 * 28 * 28)
os.environ['FPS_MAX_FRAMES'] = str(16)
os.environ['SAMPLE_MODE'] = "true"

class ModelEvaluator:
    def __init__(
        self,
        model_path: str,
        dataset_path: str,
        output_path: str,
        test_samples: Optional[int] = None,
        batch_size: int = 32,
        random_seed: int = 42,
        save_token_info: bool = False
    ):
        # Basic configurations
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.test_samples = test_samples
        self.batch_size = batch_size
        
        # Set random seeds
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        
        # Load model and processor
        self._setup_model_and_processor()
        
        # Load dataset
        self._load_data()
        
        # Question template
        self.question_template = "{Question} Output the thinking process in <think> </think> and final answer (letters separated by , if multiple) in <answer> </answer> tags. "

        # Add hyperparameters from grpo.py
        self.ANSWER_WEIGHT = 1.0
        self.THINK_WEIGHT = 0.0
        self.LOG_PROB_THRESHOLD = -10

        self.save_token_info = save_token_info

    def _setup_model_and_processor(self):
        """Set up model and processor"""
        print(f"Loading model from {self.model_path}")
        
        # Load appropriate model based on model type
        if "Qwen2-VL" in self.model_path:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        elif "Qwen2.5-VL" in self.model_path:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_path}")
            
        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.processor.tokenizer.padding_side = 'left'
        
        # Get token information for tags
        self.tag_tokens = {
            "<answer>": self.processor.tokenizer.encode("<answer>", add_special_tokens=False),
            "</answer>": self.processor.tokenizer.encode("</answer>", add_special_tokens=False),
            "<think>": self.processor.tokenizer.encode("<think>", add_special_tokens=False),
            "</think>": self.processor.tokenizer.encode("</think>", add_special_tokens=False)
        }

    def _load_data(self):
        """Load and prepare dataset"""
        print(f"Loading data from {self.dataset_path}")
        with open(self.dataset_path, "r") as f:
            all_data = json.load(f)
            
            # If test_samples is specified, randomly sample, otherwise use all data
            if self.test_samples is not None:
                self.data = random.sample(all_data, min(self.test_samples, len(all_data)))
            else:
                self.data = all_data
            
            print(f"Loaded {len(self.data)} samples for evaluation")
            
            # Ensure ground truth has <answer> tags
            for item in self.data:
                if not item['solution'].startswith("<answer>"):
                    item['solution'] = f"<answer>{item['solution']}</answer>"

    def calculate_confidence(self, start_pos, end_pos, logps):
        """Calculate confidence score for a section of tokens."""
        if start_pos != -1 and end_pos != -1 and start_pos < end_pos:
            section_logps = logps[start_pos:end_pos]
            valid_logps = torch.where(
                section_logps > self.LOG_PROB_THRESHOLD,
                section_logps,
                torch.tensor(-float('inf'), device=section_logps.device)
            )
            probs = torch.exp(valid_logps)
            return probs.mean().item()
        return 0.0

    def find_tag_positions(self, start_tokens, end_tokens, sequence):
        """Find the positions of start and end tags in the sequence."""
        start_pos = -1
        end_pos = -1
        
        # Use sequence directly as token_ids
        token_ids = sequence
        
        seq_len = token_ids.shape[0]
        end_tag_len = len(end_tokens)
        start_tag_len = len(start_tokens)
        
        for i in range(seq_len - start_tag_len + 1):
            if token_ids[i:i+start_tag_len].tolist() == start_tokens:
                start_pos = i + start_tag_len
                for j in range(start_pos, seq_len - end_tag_len + 1):
                    if token_ids[j:j+end_tag_len].tolist() == end_tokens:
                        end_pos = j
                        break
                if end_pos != -1:
                    break
        
        return start_pos, end_pos

    def extract_answer(self, output_str):
        """Extract answer content from output string"""
        try:
            # First try to extract content within <answer> tags
            content_match = re.search(r"<answer>(.*?)</answer>", output_str, re.DOTALL)
            if content_match:
                return content_match.group(1).strip()
            
            # If no tags, try to extract answer from the end
            answer_match = re.search(r'(?:answer|Answer|ANSWER)[:\s]+([A-Z](?:\s*,\s*[A-Z])*)', output_str)
            if answer_match:
                return answer_match.group(1).strip()
            
            return ""
            
        except Exception as e:
            logging.debug(f"Error extracting answer: {str(e)}")
            return ""

    def extract_think_content(self, output_str):
        """Extract thinking content from output string"""
        try:
            # First try to extract content within <think> tags
            think_match = re.search(r'<think>(.*?)</think>', output_str, re.DOTALL)
            if think_match:
                return think_match.group(1).strip()
            
            # If no tags, try to extract analysis part (all content before answer)
            answer_pos = output_str.lower().find('answer:')
            if answer_pos != -1:
                think_content = output_str[:answer_pos].strip()
                return think_content
            
            return ""
            
        except Exception as e:
            logging.debug(f"Error extracting thinking: {str(e)}")
            return ""

    def calculate_accuracy_score(self, content, solution):
        """Calculate accuracy score using the same method as GRPO training"""
        reward = 0.0
        verification_method = "none"
        
        try:
            # Extract answer from solution if it has think/answer tags
            sol_match = re.search(r"<answer>(.*?)</answer>", solution, re.DOTALL)
            ground_truth = sol_match.group(1).strip() if sol_match else solution.strip()

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
            reward = 0.0
        
        return reward

    def check_format(self, content):
        """Check format using the same method as GRPO training"""
        pattern = r"^(?!.*<think>.*<think>)(?!.*<answer>.*<answer>)<think>.*?</think>\s*<answer>.*?</answer>$"
        match = re.match(pattern, content.strip(), re.DOTALL)
        return 1.0 if match else 0.0

    def process_batch_results(self, batch_data, batch_output_text, token_logprobs, completion_ids, tag_tokens):
        """Process results for each batch"""
        batch_results = []
        
        logging.debug("Starting to process batch results")
        logging.debug(f"Batch data size: {len(batch_data)}")
        logging.debug(f"Batch output text size: {len(batch_output_text)}")
        logging.debug(f"Token logprobs size: {len(token_logprobs)}")
        logging.debug(f"Completion IDs size: {len(completion_ids)}")
        
        # Decode completions
        completions = self.processor.batch_decode(completion_ids, skip_special_tokens=True)
        completions = [[{"role": "assistant", "content": completion}] for completion in completions]
        
        # Prepare input data
        prompts = [x["problem"] for x in batch_data]
        solutions = [x.get('solution', '') for x in batch_data]
        
        # Calculate think_confidence, answer_confidence and overall_confidence
        think_confidence = []
        answer_confidence = []
        overall_confidence = []
        weighted_score = []
        
        for idx, (token_logprobs_row, completion_id_row) in enumerate(zip(token_logprobs, completion_ids)):
            # Get <think> and <answer> tokens
            think_start_tokens = tag_tokens["<think>"][:2]
            think_end_tokens = tag_tokens["</think>"][:2]
            answer_start_tokens = tag_tokens["<answer>"][:2]
            answer_end_tokens = tag_tokens["</answer>"]
            
            # Use find_tag_positions to locate <think> and <answer> positions
            think_start, think_end = self.find_tag_positions(think_start_tokens, think_end_tokens, completion_id_row)
            answer_start, answer_end = self.find_tag_positions(answer_start_tokens, answer_end_tokens, completion_id_row)
            
            # Calculate confidence scores
            think_conf = self.calculate_confidence(think_start, think_end, token_logprobs_row)
            answer_conf = self.calculate_confidence(answer_start, answer_end, token_logprobs_row)
            
            think_confidence.append(think_conf)
            answer_confidence.append(answer_conf)
        
        # Process each sample
        for idx, (input_example, output_text, completion_id) in enumerate(zip(batch_data, batch_output_text, completion_ids)):
            try:
                logging.debug(f"Processing sample: {input_example}")
                
                # Clean output text
                output_text = re.sub(r'system.*?assistant\n', '', output_text, flags=re.DOTALL)
                output_text = output_text.strip()
                
                # Extract answer and thinking content
                model_answer = self.extract_answer(output_text)
                think_content = self.extract_think_content(output_text)
                ground_truth = input_example.get('solution', '')
                
                # Calculate accuracy and format scores
                accuracy_score = self.calculate_accuracy_score(model_answer, ground_truth)
                format_score = self.check_format(output_text)

                # Store results
                result = {
                    'question': input_example,
                    'ground_truth': ground_truth,
                    'model_output': output_text,
                    'extracted_answer': model_answer,
                    'think_content': think_content,
                    'accuracy_score': float(accuracy_score),
                    'format_score': float(format_score),
                    'think_confidence': float(think_confidence[idx]),
                    'answer_confidence': float(answer_confidence[idx]),
                    'total_tokens_generated': len(completion_id),
                }
                batch_results.append(result)
                
                # Log results
                logging.debug(f"Processed result: {result}")
                
            except Exception as e:
                logging.debug(f"Error processing sample: {str(e)}")
                continue

        if not batch_results:
            logging.warning("No results were processed. Check input data and processing logic.")
        
        logging.debug("Finished processing batch results")
        
        # Create model name directory
        model_name = os.path.basename(self.model_path)
        output_dir = os.path.join("results", model_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results to JSON file
        output_file_path = os.path.join(output_dir, "batch_results.json")
        with open(output_file_path, 'w') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Batch results saved to {output_file_path}")
        
        return batch_results

    def evaluate(self):
        """Execute evaluation"""
        logging.info("Starting evaluation...")
        all_outputs = []
        
        # Initialize counters for single and multiple choice questions
        single_choice_results = []
        multiple_choice_results = []
        
        # Get total number of samples
        total_samples = len(self.data)
        
        # Create temporary file path
        temp_output_path = self.output_path.replace('.json', '_temp.json')
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # If temporary file exists, try to resume previous progress
        if os.path.exists(temp_output_path):
            try:
                with open(temp_output_path, 'r') as f:
                    saved_data = json.load(f)
                    all_outputs = saved_data.get('results', [])
                    completed_samples = len(all_outputs)
                    print(f"Resuming from {completed_samples} completed samples")
            except:
                print("Could not load previous progress, starting fresh")
                completed_samples = 0
        else:
            completed_samples = 0

        # Prepare messages with system prompt
        SYSTEM_PROMPT = (
            "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
            "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
            "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
            "<think> reasoning process here </think><answer> answer here </answer>"
        )

        messages = []
        for item in self.data:
            is_multiple = "," in item["solution"]
            question_type = "multiple choice question" if is_multiple else "single choice question"
            message = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text":self.question_template.format(Question=item['problem'])}
                    ]
                }
            ]
            messages.append(message)

        # Process batches
        for i in tqdm(range(0, total_samples, self.batch_size)):
            batch_messages = messages[i:i + self.batch_size]
            batch_data = self.data[i:i + self.batch_size]
            
            # Prepare inputs
            text = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
                   for msg in batch_messages]
            
            # Process video inputs
            formatted_messages = []
            video_paths = []
            for item in batch_data:
                formatted_message = [{
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": self.question_template.format(Question=item['problem'])},
                    ],
                }]
                formatted_messages.append(formatted_message)
                video_paths.append(item['video'])

            batch_video_inputs = []
            for msg, video_path in zip(formatted_messages, video_paths):
                msg[0]['content'][0]['text'] = video_path
                batch_video_inputs.append(process_vision_info(msg)[0])

            # Generate outputs
            inputs = self.processor(
                text=text,
                videos=batch_video_inputs,
                padding=True,
                return_tensors="pt",
            )

            # Move input data to the same device as the model
            device = next(self.model.parameters()).device  # Get the device where model is located
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v  # Only move Tensor type data
                for k, v in inputs.items()
            }

            # Modify generation config to match grpo_trainer.py
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    use_cache=True,
                    max_new_tokens=4096,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.001,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True
                )

            # Decode generated text
            batch_output_text = self.processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)
            
            # Get log probabilities for each token
            token_logprobs = self.get_per_token_logps(self.model, generated_ids.sequences)
            
            # Remove log probabilities for prompt part
            prompt_length = inputs["input_ids"].size(1)
            token_logprobs = token_logprobs[:, prompt_length - 1 :]
            
            # Get completion_ids
            completion_ids = generated_ids.sequences[:, prompt_length:]
            
            # Process batch results
            batch_results = self.process_batch_results(
                batch_data, 
                batch_output_text, 
                token_logprobs,  # per_token_logps
                completion_ids,  # completion_ids
                self.tag_tokens  # tag_tokens
            )
            all_outputs.extend(batch_results)

            # Calculate and save current metrics in real-time
            if len(all_outputs) > 0:  # Ensure all_outputs is not empty
                # Separate results by question type
                for result in batch_results:
                    if "," in result['ground_truth']:
                        multiple_choice_results.append(result)
                    else:
                        single_choice_results.append(result)

                # Calculate metrics for each type
                single_metrics = {
                    'strict_accuracy': sum(1 for r in single_choice_results if r['accuracy_score'] == 1.0) / len(single_choice_results) * 100 if single_choice_results else 0,
                    'partial_accuracy': sum(r['accuracy_score'] for r in single_choice_results) / len(single_choice_results) * 100 if single_choice_results else 0,
                    'count': len(single_choice_results)
                }
                
                multiple_metrics = {
                    'strict_accuracy': sum(1 for r in multiple_choice_results if r['accuracy_score'] == 1.0) / len(multiple_choice_results) * 100 if multiple_choice_results else 0,
                    'partial_accuracy': sum(r['accuracy_score'] for r in multiple_choice_results) / len(multiple_choice_results) * 100 if multiple_choice_results else 0,
                    'count': len(multiple_choice_results)
                }

                current_metrics = {
                    'overall': {
                        'strict_accuracy': sum(1 for r in batch_results if r['accuracy_score'] == 1.0) / len(batch_results) * 100,
                        'partial_accuracy': sum(r['accuracy_score'] for r in batch_results) / len(batch_results) * 100,
                    },
                    'single_choice': single_metrics,
                    'multiple_choice': multiple_metrics,
                    'avg_think_confidence': sum(r['think_confidence'] for r in batch_results) / len(batch_results),
                    'avg_answer_confidence': sum(r['answer_confidence'] for r in batch_results) / len(batch_results),
                    'completed_samples': len(batch_results),
                    'total_samples': total_samples,
                    'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            else:
                current_metrics = {
                    'overall': {
                        'strict_accuracy': 0.0,
                        'partial_accuracy': 0.0,
                    },
                    'single_choice': {'strict_accuracy': 0.0, 'partial_accuracy': 0.0, 'count': 0},
                    'multiple_choice': {'strict_accuracy': 0.0, 'partial_accuracy': 0.0, 'count': 0},
                    'avg_think_confidence': 0.0,
                    'avg_answer_confidence': 0.0,
                    'completed_samples': 0,
                    'total_samples': total_samples,
                    'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

            # Save temporary file
            self.save_results(temp_output_path, batch_results, current_metrics)

            # Update progress printing to include separate metrics
            if (i + self.batch_size) % (self.batch_size * 10) == 0:  # Print every 10 batches
                print(f"\nCurrent metrics at {len(all_outputs)}/{total_samples} samples:")
                print("Overall metrics:")
                print(f"  Strict Accuracy: {current_metrics['overall']['strict_accuracy']:.2f}%")
                print(f"  Partial Accuracy: {current_metrics['overall']['partial_accuracy']:.2f}%")
                print("\nSingle Choice Questions:")
                print(f"  Count: {current_metrics['single_choice']['count']}")
                print(f"  Strict Accuracy: {current_metrics['single_choice']['strict_accuracy']:.2f}%")
                print(f"  Partial Accuracy: {current_metrics['single_choice']['partial_accuracy']:.2f}%")
                print("\nMultiple Choice Questions:")
                print(f"  Count: {current_metrics['multiple_choice']['count']}")
                print(f"  Strict Accuracy: {current_metrics['multiple_choice']['strict_accuracy']:.2f}%")
                print(f"  Partial Accuracy: {current_metrics['multiple_choice']['partial_accuracy']:.2f}%")

        # After evaluation completes, rename temporary file to final file
        if os.path.exists(temp_output_path):
            os.replace(temp_output_path, self.output_path)
            print(f"\nResults saved to {self.output_path}")
        
        return current_metrics, all_outputs

    def get_per_token_logps(self, model, input_ids):
        """
        Calculate log probabilities for each token
        Implementation referenced from grpo_trainer.py
        Args:
            model: Model being used
            input_ids: Input token ids
        Returns:
            per_token_logps: Log probabilities for each token, shape (B, L-1)
        """
        # Get logits
        with torch.no_grad():
            logits = model(input_ids).logits  # (B, L, V)
        
        # Remove last logit as it corresponds to next token prediction
        logits = logits[:, :-1, :]  # (B, L-1, V)
        
        # Remove first input_id as we don't have its logits
        input_ids = input_ids[:, 1:]  # (B, L-1)
        
        # Calculate log probabilities for each token
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            # Calculate log probability distribution
            log_probs = logits_row.log_softmax(dim=-1)  # (L-1, V)
            
            # Extract log probabilities for corresponding tokens
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)  # (L-1,)
            per_token_logps.append(token_log_prob)
        
        # Stack results into tensor
        return torch.stack(per_token_logps)  # (B, L-1)

    def save_results(self, temp_output_path, batch_results, current_metrics):
        # Open file in append mode
        with open(temp_output_path, 'a', encoding='utf-8') as f:
            # Save current batch results
            for result in batch_results:
                # Remove token-related data if not saving token info
                if not self.save_token_info:
                    result.pop('total_tokens_generated', None)
                    # Remove other token-related data here if needed

                # Write each result on a new line
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        # Save current metrics to separate file
        metrics_path = temp_output_path.replace('.json', '_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(current_metrics, f, indent=2, ensure_ascii=False)

def get_latest_checkpoint(model_base_path):
    """
    Get the latest checkpoint from the model directory
    Args:
        model_base_path: Base path of the model directory
    Returns:
        str: Full path to the latest checkpoint
    """
    try:
        # Find all checkpoint directories
        checkpoint_pattern = os.path.join(model_base_path, "checkpoint-*")
        checkpoints = glob.glob(checkpoint_pattern)

        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {model_base_path}")

        # Extract and sort checkpoint numbers
        checkpoint_numbers = []
        for checkpoint in checkpoints:
            match = re.search(r'checkpoint-(\d+)$', checkpoint)
            if match:
                checkpoint_numbers.append((int(match.group(1)), checkpoint))

        # Sort by checkpoint number and return the latest one
        latest_checkpoint = sorted(checkpoint_numbers, key=lambda x: x[0])[-1][1]
        return latest_checkpoint

    except Exception as e:
        logging.error(f"Error finding latest checkpoint: {str(e)}")
        raise

def evaluate(model_name, batch_size):
    """
    Evaluate model performance
    Args:
        model_name: Name/path of the model to evaluate
        batch_size: Initial batch size for evaluation
    Returns:
        tuple: (metrics, results) evaluation metrics and detailed results
    """
    try:
        # Configure parameters
        os.environ['DEBUG_MODE'] = "true"
        private_data_root = "data/CLEVRER"
        dataset_name = "clevrer_counterfactual_val"
        
        # Construct model path
        model_path = os.path.join(private_data_root, model_name)
        base_path = model_path.split('/checkpoint-')[0]
        
        # If specified checkpoint doesn't exist, try to find the latest one
        if not os.path.exists(model_path):
            logging.warning(f"Specified model path not found: {model_path}")
            # Get base path (without checkpoint part)
            base_path = model_path.split('/checkpoint-')[0]
            if os.path.exists(base_path):
                model_path = get_latest_checkpoint(base_path)
                logging.info(f"Using latest checkpoint instead: {model_path}")
            else:
                raise FileNotFoundError(f"Model base path not found: {base_path}")
        
        dataset_path = os.path.join(private_data_root, "CLEVRER", f"{dataset_name}.json")
        output_path = os.path.join(".", "logs", dataset_name, "test", base_path, f"{os.path.basename(model_path)}_eval.json")
        
        # Check if dataset exists
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
            
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        original_batch_size = batch_size
        current_batch_size = original_batch_size

        while True:
            try:
                # Initialize evaluator with current batch size
                evaluator = ModelEvaluator(
                    model_path=model_path,
                    dataset_path=dataset_path,
                    output_path=output_path,
                    batch_size=current_batch_size,
                    save_token_info=False,
                    test_samples=None,
                )
                
                return evaluator.evaluate()

            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                if "out of memory" in str(e) or "CUDA" in str(e):
                    
                    # Clear GPU cache
                    torch.cuda.empty_cache()
                    
                    # If batch_size is already 1, cannot reduce further
                    if current_batch_size == 1:
                        logging.error("Cannot reduce batch size further. Already at minimum (1).")
                        raise e
                    
                    # Reduce batch_size
                    current_batch_size = max(1, current_batch_size // 2)
                    logging.warning(f"GPU OOM error. Reducing batch size to {current_batch_size} and retrying...")
                    continue
                else:
                    raise e
        
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        raise

def main():
    """
    Main function to handle command line arguments and run evaluation
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                      help="Name/path of the model to evaluate")
    parser.add_argument("--batch_size", type=int, default=16,
                      help="Batch size for evaluation (default: 16)")
    args = parser.parse_args()
    
    try:
        metrics, results = evaluate(model_name=args.model_name, batch_size=args.batch_size)
        print(f"Evaluation completed. Metrics: {metrics}")
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

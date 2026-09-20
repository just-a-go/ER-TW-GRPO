import json
import random

def negate_question(question):
    # Convert to lowercase for uniform processing
    question = question.lower()
    
    # Define negation mapping
    negation_map = {
        "did": "didn't",
        "does": "doesn't",
        "do": "don't",
        "is": "isn't",
        "are": "aren't",
        "was": "wasn't",
        "were": "weren't"
    }
    
    # Tokenize the question
    words = question.split()
    
    # Handle special cases at the beginning of the question
    if len(words) >= 2:
        # If the second word is in the negation map, replace it
        if words[1] in negation_map:
            words[1] = negation_map[words[1]]
            return " ".join(words)
    
    return question

def negate_answer(options, original_answer, num_keep=None):
    """
    Invert the answer and rearrange options.
    In the negated question:
    - The originally correct option becomes incorrect
    - All originally incorrect options become correct
    params:
        options: list of options
        original_answer: original answer (e.g., "(A)")
        num_keep: number of incorrect options to keep, if None then random
    """
    # Get the index and content of the original correct answer
    correct_idx = ord(original_answer[1]) - ord('A')
    correct_option = options[correct_idx]
    
    # Get the list of wrong options (these become correct in the negated question)
    wrong_options = [opt for i, opt in enumerate(options) if i != correct_idx]
    
    # Randomly decide how many correct options (originally wrong) to keep
    if num_keep is None:
        num_keep = random.randint(1, len(wrong_options))
    
    # Randomly select the correct options to keep
    kept_options = random.sample(wrong_options, num_keep)
    
    # Create new option list: original correct answer (now wrong) + kept correct options
    new_options = [correct_option] + kept_options
    random.shuffle(new_options)  # Shuffle the order
    
    # Relabel options as (A), (B), (C)...
    relabeled_options = []
    for i, opt in enumerate(new_options):
        content = opt[4:] if opt.startswith("(") else opt
        relabeled_options.append(f"({chr(ord('A') + i)}) {content}")
    
    # Find the new position of the original correct answer (now wrong), all others are correct
    correct_content = correct_option[4:] if correct_option.startswith("(") else correct_option
    all_answers = []
    for i, opt in enumerate(new_options):
        content = opt[4:] if opt.startswith("(") else opt
        if content != correct_content:
            all_answers.append(f"({chr(ord('A') + i)})")
    
    # Return all correct answers, comma-separated
    new_answer = ",".join(all_answers)
    
    return relabeled_options, new_answer

def format_question_with_options(question, options):
    """
    Format the question and options into a single string
    """
    # Remove the option markers (parentheses)
    formatted_options = []
    for opt in options:
        if opt.startswith("("):
            # Remove the leading "(X) " format
            formatted_options.append(opt[4:])
        else:
            formatted_options.append(opt)
    
    # Combine question and options
    option_texts = [f"{chr(ord('A') + i)} {opt}" for i, opt in enumerate(formatted_options)]
    return question + "\n" + "\n".join(option_texts)

def format_answer(answer):
    """
    Format the answer as "<answer>X,Y,Z</answer>"
    """
    # Remove parentheses, keep only letters
    answer_letters = [a[1] for a in answer.split(",")]
    return f"<answer>{','.join(answer_letters)}</answer>"

def convert_video_path(old_path):
    """
    Convert old video path format to new format
    old_path: "1106/4260763967.mp4"
    new_path: "/NExTQA/videos/4260763967.mp4"
    """
    # Extract video ID from old path
    video_id = old_path.split('/')[-1].replace('.mp4', '')
    # Build new path
    return f"/NExTQA/videos/{video_id}.mp4"

def main():
    # Read JSON file
    input_path = 'evaluation/nextgqa_val.json'
    output_path = 'evaluation/nextgqa_val_mixed.json'
    
    print(f"Reading from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} questions...")
    # Process each question
    augmented_data = []
    for item in data:
        # Copy original data
        processed_item = item.copy()
        
        # Convert video path
        processed_item['video'] = convert_video_path(item['video'])
        
        # Randomly decide whether to negate (50% chance)
        if random.random() < 0.5:
            # Negate the question
            negated_question = negate_question(item['question'])
            
            # Invert answer and rearrange options
            new_options, new_answer = negate_answer(
                item['options'], 
                item['answer']
            )
            
            # Format question and options
            processed_item['problem'] = format_question_with_options(negated_question, new_options)
            # Format answer
            processed_item['solution'] = format_answer(new_answer)
            
            # Update other fields
            processed_item['question'] = negated_question
            processed_item['options'] = new_options
            processed_item['answer'] = new_answer
            processed_item['is_negated'] = True
        else:
            # Keep original, just reformat
            processed_item['problem'] = format_question_with_options(item['question'], item['options'])
            processed_item['solution'] = format_answer(item['answer'])
            processed_item['is_negated'] = False
        
        augmented_data.append(processed_item)
    
    print(f"Saving to {output_path}")
    # Save modified JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, indent=4, ensure_ascii=False)
    
    print(f"Done! Processed {len(augmented_data)} questions.")
    print(f"Negated questions: {sum(1 for item in augmented_data if item['is_negated'])}")
    print(f"Original questions: {sum(1 for item in augmented_data if not item['is_negated'])}")

if __name__ == "__main__":
    main()
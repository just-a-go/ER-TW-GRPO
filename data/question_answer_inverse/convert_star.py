import json
import random

def negate_question(question):
    # Convert question to lowercase for uniform processing
    question = question.lower()
    
    # Define negation mapping
    negation_map = {
        "did": "didn't",
        "was": "wasn't",
        "able": "unable",
        "would": "wouldn't",
        "what will": "what won't",
        "what happened": "what didn't happen"
    }
    
    # First check for full phrase matches
    for phrase, negation in negation_map.items():
        if question.startswith(phrase):  # Changed to use startswith instead of in
            return question.replace(phrase, negation, 1)  # Only replace first occurrence
    
    # Token processing
    words = question.split()
    
    # Handle special cases at the beginning of the question
    if len(words) >= 2:
        # Check first two words as a phrase
        first_two = " ".join(words[:2])
        if first_two in negation_map:
            return negation_map[first_two] + " " + " ".join(words[2:])
        # If second word is in negation map, replace it
        if words[1] in negation_map:
            words[1] = negation_map[words[1]]
            return " ".join(words)
    
    return question

def negate_answer(options, original_answer, num_keep=None):
    """
    Invert answer and rearrange options.
    In negated questions:
    - Originally correct options become incorrect
    - Originally incorrect options become correct
    params:
        options: list of options
        original_answer: original answer (e.g. "<answer>B</answer>")
        num_keep: number of incorrect options to keep, if None then random
    """
    # Extract answer letter from <answer>X</answer> format
    correct_letter = original_answer.replace('<answer>', '').replace('</answer>', '')
    correct_idx = ord(correct_letter) - ord('A')
    correct_option = options[correct_idx]
    
    # Get list of incorrect options (these become correct in negated question)
    wrong_options = [opt for i, opt in enumerate(options) if i != correct_idx]
    
    # Randomly decide how many correct options (originally incorrect) to keep
    if num_keep is None:
        num_keep = random.randint(1, len(wrong_options))
    
    # Randomly select correct options to keep
    kept_options = random.sample(wrong_options, num_keep)
    
    # Create new option list: original correct answer (now incorrect) + kept correct options
    new_options = [correct_option] + kept_options
    random.shuffle(new_options)  # Randomize order
    
    # Find new position of original correct answer (now incorrect), all other options are correct
    all_answers = []
    for i, opt in enumerate(new_options):
        if opt != correct_option:
            all_answers.append(chr(ord('A') + i))
    
    # Return new options and formatted answer
    return new_options, f"<answer>{','.join(all_answers)}</answer>"

def format_question_with_options(question, options):
    """
    Format question and options into a single string
    """
    # Remove bracket markers from options
    formatted_options = []
    for opt in options:
        if opt.startswith("("):
            # Remove leading "(X) " format
            formatted_options.append(opt[4:])
        else:
            formatted_options.append(opt)
    
    # Combine question and options
    option_texts = [f"{chr(ord('A') + i)} {opt}" for i, opt in enumerate(formatted_options)]
    return question + "\n" + "\n".join(option_texts)

def format_answer(answer):
    """
    Format answer as "<answer>X,Y,Z</answer>"
    """
    # Remove brackets, keep only letters
    answer_letters = [a[1] for a in answer.split(",")]
    return f"<answer>{','.join(answer_letters)}</answer>"

def convert_video_path(old_path):
    """
    Convert original video path to new format
    old_path: "1106/4260763967.mp4"
    new_path: "/NExTQA/videos/4260763967.mp4"
    """
    # Extract video ID from original path
    video_id = old_path.split('/')[-1].replace('.mp4', '')
    # Construct new path
    return f"/NExTQA/videos/{video_id}.mp4"

def convert_star_format(input_file, output_file):
    # Read original json file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted_data = []
    
    for item in data:
        # Only process STAR dataset data
        if item.get('data_source') == 'STAR':
            # Build new format
            new_item = {
                'id': str(item['problem_id']),  # Convert problem_id to string as id
                'video': '/STAR/' + item['path'].split('/')[-1],  # Modify path format
                'question': item['problem'],
                'options': [opt.split('. ')[1] for opt in item['options']],  # Remove option letter markers
                'problem': item['problem'] + '\n' + '\n'.join(item['options']),  # Combine question and options
                'solution': item['solution'],
                'type': 'MC'  # Assume all STAR dataset are multiple choice
            }
            converted_data.append(new_item)
    
    # Write new json file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=4, ensure_ascii=False)

def main():
    # Read JSON file
    # input_path = 'Annotations/NextGQA/Video-R1-260k.json'
    star_path = 'evaluation/STAR.json'
    
    # print(f"Reading from {input_path}")
    # convert_star_format(input_path, star_path)
    input_path = star_path
    output_path = 'evaluation/STAR_mixed.json'
    
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
        
        # Randomly decide whether to negate (50% probability)
        if random.random() < 0.5:
            # Negate question
            negated_question = negate_question(item['question'])
            
            # Invert answer and rearrange options
            new_options, new_answer = negate_answer(
                item['options'], 
                item['solution']  # Use solution instead of answer
            )
            
            # Format question and options
            processed_item['problem'] = format_question_with_options(negated_question, new_options)
            processed_item['solution'] = new_answer
            
            # Update other fields
            processed_item['question'] = negated_question
            processed_item['options'] = new_options
            processed_item['is_negated'] = True
        else:
            # Keep original, just reformat
            processed_item['problem'] = format_question_with_options(item['question'], item['options'])
            processed_item['is_negated'] = False
        
        augmented_data.append(processed_item)
    
    print(f"Saving to {output_path}")
    # Save modified JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, indent=4, ensure_ascii=False)
    
    print(f"Done! Processed {len(augmented_data)} questions.")
    print(f"Negated questions: {sum(1 for item in augmented_data if item['is_negated'])}")
    print(f"Original questions: {sum(1 for item in augmented_data if not item['is_negated'])}")
    
    print(f"Done! Processed questions.")

if __name__ == "__main__":
    main()
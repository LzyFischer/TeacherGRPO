import json
import jsonlines

def merge_datasets(original_jsonl_path, generated_json_path, output_jsonl_path):
    """
    Merge original dataset with generated responses.
    
    Args:
        original_jsonl_path: Path to original JSONL file
        generated_json_path: Path to generated JSON file with responses
        output_jsonl_path: Path to save merged JSONL file
    """
    # Load original dataset
    original_data = []
    with jsonlines.open(original_jsonl_path) as reader:
        for obj in reader:
            original_data.append(obj)
    
    # Load generated responses
    with open(generated_json_path, 'r') as f:
        generated_data = json.load(f)
    
    # Create mapping from question to generated response
    # Using question as key since pid might not be reliable
    question_to_generated = {}
    for detail in generated_data['details']:
        # Extract question from prompt (remove the instruction part)
        prompt = detail['prompt']
        question = prompt.replace('Question: ', '').replace('\n\nPlease reason step by step, and put your final answer within \\boxed{}.', '').strip()
        question_to_generated[question] = detail['generated']
    
    # Merge datasets
    merged_data = []
    matched_count = 0
    unmatched_count = 0
    
    for original in original_data:
        merged_entry = original.copy()
        
        # Try to find matching generated response
        if original['question'] in question_to_generated:
            merged_entry['response'] = question_to_generated[original['question']]
            matched_count += 1
        else:
            # Keep original response if no match found
            unmatched_count += 1
            print(f"Warning: No generated response found for question: {original['question'][:50]}...")
        
        merged_data.append(merged_entry)
    
    # Write merged data to output file
    with jsonlines.open(output_jsonl_path, mode='w') as writer:
        writer.write_all(merged_data)
    
    # Print statistics
    print(f"\nMerge Statistics:")
    print(f"Total original entries: {len(original_data)}")
    print(f"Total generated responses: {len(generated_data['details'])}")
    print(f"Matched entries: {matched_count}")
    print(f"Unmatched entries: {unmatched_count}")
    print(f"Output saved to: {output_jsonl_path}")

# Usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge original dataset with generated responses.")
    parser.add_argument("--original_file", type=str, required=True, help="Path to original JSONL file")
    parser.add_argument("--generated_file", type=str, required=True, help="Path to generated JSON file with responses")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save merged JSONL file")
    args = parser.parse_args()
    merge_datasets(args.original_file, args.generated_file, args.output_file)

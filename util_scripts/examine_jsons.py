import json
import jsonlines
import argparse
from pathlib import Path

def prettify_json(json_data):
    return json.dumps(json_data, indent=2)

def save_samples_as_jsonl(samples, output_file):
    with open(output_file, 'w') as f:
        for sample in samples:
            json.dump(sample, f)
            f.write('\n')

def examine_and_save_json_file(file_path, output_dir, num_samples):
    if file_path.suffix == ".jsonl":
        data = []
        with jsonlines.open(file_path) as reader:
            for sample in reader:
                if len(data) >= num_samples:
                    break
                data.append(sample)
    else:
        with open(file_path, 'r') as file:
            data = json.load(file)
    
    if isinstance(data, list):
        samples = data[:num_samples]
    elif isinstance(data, dict):
        samples = list(data.items())[:num_samples]
    else:
        raise ValueError("Unsupported JSON structure. Expected a list or dictionary.")

    print(f"\nExamining {file_path}:")
    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i}:")
        print(prettify_json(sample))
    
    output_file = output_dir / 'sample.jsonl'
    save_samples_as_jsonl(samples, output_file)
    print(f"Samples saved to {output_file}")

def process_directory(root_dir, output_root, num_samples):
    root_path = Path(root_dir)
    output_root_path = Path(output_root)
    
    if not root_path.is_dir():
        print(f"Error: '{root_dir}' is not a valid directory.")
        return

    for subdir in root_path.iterdir():
        if subdir.is_dir():
            train_file = subdir / 'train.json'
            alternate_train_file = subdir / 'train.jsonl'
            if train_file.exists() or alternate_train_file.exists():
                relative_path = subdir.relative_to(root_path)
                output_dir = output_root_path / relative_path
                output_dir.mkdir(parents=True, exist_ok=True)
                
                try:
                    existing_train_file = train_file if train_file.exists() else alternate_train_file
                    examine_and_save_json_file(existing_train_file, output_dir, num_samples)
                except json.JSONDecodeError:
                    print(f"Error: '{train_file}' is not a valid JSON file.")
                except Exception as e:
                    print(f"An error occurred while processing '{train_file}': {str(e)}")
            else:
                print(f"Warning: No 'train.json' found in {subdir}")

def main():
    parser = argparse.ArgumentParser(description="Examine, prettify, and save samples from JSON datasets as JSONL")
    parser.add_argument("root_dir", type=str, help="Root directory containing subdirectories with JSON files")
    parser.add_argument("output_dir", type=str, help="Output root directory for saving samples")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to examine and save from each file (default: 5)")
    args = parser.parse_args()

    process_directory(args.root_dir, args.output_dir, args.samples)

if __name__ == "__main__":
    main()

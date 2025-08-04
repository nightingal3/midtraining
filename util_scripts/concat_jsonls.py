import argparse
import json
import os

from tqdm import tqdm


def concatenate_jsonl_files(root_dir, output_file, test_mode=False, test_lines=10000):
    total_files = sum(
        len(files)
        for _, _, files in os.walk(root_dir)
        if any(f.endswith(".jsonl") for f in files)
    )

    with open(output_file, "w") as outfile:
        with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
            for subdir, _, files in os.walk(root_dir):
                if "concat" in subdir:
                    continue
                for file in files:
                    if file.endswith(".jsonl"):
                        file_path = os.path.join(subdir, file)
                        lines_processed = 0
                        total_lines = (
                            test_lines
                            if test_mode
                            else sum(1 for _ in open(file_path, "r"))
                        )

                        with tqdm(
                            total=total_lines,
                            desc=f"Processing {file}",
                            unit="line",
                            leave=False,
                        ) as file_pbar:
                            with open(file_path, "r") as infile:
                                for line in infile:
                                    line = line.strip()
                                    if not line:
                                        continue

                                    try:
                                        # Ensure the line is valid JSON
                                        json_object = json.loads(line)
                                        json.dump(json_object, outfile)
                                        outfile.write(
                                            "\n"
                                        )  # Add newline after each JSON object
                                        lines_processed += 1

                                    except json.JSONDecodeError:
                                        print(f"Skipping invalid JSON in {file_path}")

                                    file_pbar.update(1)

                                    if test_mode and lines_processed >= test_lines:
                                        break

                        pbar.set_postfix({"Lines processed": lines_processed})
                        pbar.update(1)

                        # No need to add an extra newline here, as we're already adding one after each JSON object


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate JSONL files")
    parser.add_argument(
        "--root_directory",
        help="Root directory containing JSONL files",
        default="../manifold/all_in_one_pretraining/datasets/knowledgeqa_formatted",
    )
    parser.add_argument(
        "--output_file",
        help="Output file name",
        default="../manifold/all_in_one_pretraining/datasets/knowledgeqa_formatted/concat_new.jsonl",
    )
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument(
        "--test_lines",
        type=int,
        default=10000,
        help="Number of lines to process per file in test mode",
    )
    args = parser.parse_args()

    concatenate_jsonl_files(
        args.root_directory,
        args.output_file,
        test_mode=args.test,
        test_lines=args.test_lines,
    )
    print(f"Concatenation complete. Output saved to {args.output_file}")

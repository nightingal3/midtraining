#!/usr/bin/env python3

import re

def extract_timing_from_checkpoint(checkpoint_dir):
    """Extract timing step from checkpoint directory path."""
    if not isinstance(checkpoint_dir, str):
        return None
    
    # Look for patterns like "from_50k", "from_100k", etc.
    timing_pattern = r'from_(\d+)k'
    match = re.search(timing_pattern, checkpoint_dir)
    if match:
        return f"from_{match.group(1)}k"
    return None

def sort_timing_steps(timing_steps):
    """Sort timing steps numerically."""
    timing_tuples = []
    other_steps = []
    
    for step in timing_steps:
        if step.startswith("from_") and step.endswith("k"):
            try:
                step_num = int(step[5:-1])  # Extract number from "from_50k"
                timing_tuples.append((step_num, step))
            except (IndexError, ValueError):
                other_steps.append(step)
        else:
            other_steps.append(step)
    
    timing_tuples.sort(key=lambda x: x[0])
    other_steps.sort()
    
    return [step for _, step in timing_tuples] + other_steps

# Test cases
test_cases = [
    '/projects/bfcu/mliu7/all_in_one_pretrainingpretrained_chkpts/ablations/pythia_160m_sc_from_50k/final',
    '/path/to/pythia_70m_sc_from_100k/step-00025000',
    '/path/to/pythia_70m_sc_from_25k/final',
    '/invalid/path/without/timing',
    None
]

print("Testing timing extraction:")
for test in test_cases:
    result = extract_timing_from_checkpoint(test)
    print(f"{test} -> {result}")

print("\nTesting timing sorting:")
timing_steps = ["from_100k", "from_25k", "from_50k", "final", "from_10k"]
sorted_steps = sort_timing_steps(timing_steps)
print(f"Original: {timing_steps}")
print(f"Sorted: {sorted_steps}")

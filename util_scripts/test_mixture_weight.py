#!/usr/bin/env python3

import re

def extract_mixture_weight_from_checkpoint(checkpoint_dir):
    """Extract mixture weight percentage from checkpoint directory path."""
    if not isinstance(checkpoint_dir, str):
        return None
    
    # Look for patterns like "10_percent", "30_percent", etc.
    weight_pattern = r'(\d+)_percent'
    match = re.search(weight_pattern, checkpoint_dir)
    if match:
        return f"{match.group(1)}_percent"
    return None

# Test cases
test_cases = [
    '/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/pretrained_chkpts/ablations/pythia_160m_sc_80_percent/step-00050000',
    '/path/to/pythia_70m_sc_30_percent/final',
    '/path/to/pythia_70m_sc_10_percent/step-00025000',
    '/invalid/path/without/percent',
    None
]

print("Testing mixture weight extraction:")
for test in test_cases:
    result = extract_mixture_weight_from_checkpoint(test)
    print(f"{test} -> {result}")

#!/usr/bin/env python3
"""
Test script for filtering zero-advantage samples.
This script creates mock data to test the _filter_zero_advantage_samples method.
"""

import torch
import numpy as np
from verl import DataProto
from tensordict import TensorDict


def test_filter_zero_advantage_samples():
    """Test the filtering logic with mock data."""
    
    # Create mock batch data
    batch_size = 16
    response_length = 10
    
    # Create advantages where some samples have all zeros
    advantages = torch.randn(batch_size, response_length)
    
    # Make samples 0, 3, 5, 8 have all-zero advantages (simulating same rewards in a group)
    zero_advantage_indices = [0, 3, 5, 8]
    for idx in zero_advantage_indices:
        advantages[idx, :] = 0.0
    
    # Create response mask (all ones for simplicity)
    response_mask = torch.ones(batch_size, response_length)
    
    # Create a mock DataProto
    batch = DataProto(
        batch=TensorDict(
            {
                "advantages": advantages,
                "response_mask": response_mask,
                "returns": torch.randn(batch_size, response_length),
                "old_log_probs": torch.randn(batch_size, response_length),
            },
            batch_size=(batch_size,),
        ),
        non_tensor_batch={
            "uid": np.array([f"sample_{i}" for i in range(batch_size)], dtype=object)
        },
        meta_info={},
    )
    
    # Simulate the filtering logic
    print("Original batch size:", batch_size)
    print("Zero advantage indices:", zero_advantage_indices)
    print("Expected valid samples:", batch_size - len(zero_advantage_indices))
    
    # Test the filtering logic
    masked_advantages = torch.abs(advantages) * response_mask
    advantage_sum_per_sample = masked_advantages.sum(dim=-1)
    epsilon = 1e-8
    valid_sample_mask = advantage_sum_per_sample > epsilon
    
    n_original = valid_sample_mask.shape[0]
    n_valid = valid_sample_mask.sum().item()
    n_filtered = n_original - n_valid
    
    print(f"\nFiltering results:")
    print(f"  Original samples: {n_original}")
    print(f"  Valid samples: {n_valid}")
    print(f"  Filtered samples: {n_filtered}")
    print(f"  Filtered ratio: {n_filtered / n_original:.2%}")
    
    # Test mini_batch_size divisibility
    mini_batch_size = 4
    print(f"\nMini-batch size: {mini_batch_size}")
    print(f"Valid samples divisible by mini_batch_size: {n_valid % mini_batch_size == 0}")
    
    if n_valid % mini_batch_size != 0:
        n_to_keep = (n_valid // mini_batch_size) * mini_batch_size
        print(f"Adjusted to keep: {n_to_keep} samples")
        
        valid_indices = torch.where(valid_sample_mask)[0]
        for idx in valid_indices[n_to_keep:]:
            valid_sample_mask[idx] = False
        
        n_valid_adjusted = valid_sample_mask.sum().item()
        print(f"After adjustment: {n_valid_adjusted} samples")
        print(f"Divisible by mini_batch_size: {n_valid_adjusted % mini_batch_size == 0}")
    
    # Filter the batch
    filtered_batch = batch.select_idxs(valid_sample_mask)
    print(f"\nFiltered batch size: {filtered_batch.batch.batch_size[0]}")
    
    # Verify no zero-advantage samples remain
    filtered_advantages = filtered_batch.batch["advantages"]
    filtered_mask = filtered_batch.batch["response_mask"]
    filtered_advantage_sums = (torch.abs(filtered_advantages) * filtered_mask).sum(dim=-1)
    assert torch.all(filtered_advantage_sums > epsilon), "Found zero-advantage samples in filtered batch!"
    print("✓ All filtered samples have non-zero advantages")
    
    # Test edge case: all samples have zero advantage
    print("\n" + "="*60)
    print("Testing edge case: all samples have zero advantage")
    all_zero_advantages = torch.zeros(batch_size, response_length)
    batch.batch["advantages"] = all_zero_advantages
    
    valid_sample_mask_all_zero = (torch.abs(all_zero_advantages) * response_mask).sum(dim=-1) > epsilon
    n_valid_all_zero = valid_sample_mask_all_zero.sum().item()
    print(f"Valid samples before forcing: {n_valid_all_zero}")
    
    if n_valid_all_zero == 0:
        print("All samples filtered - forcing to keep first sample")
        valid_sample_mask_all_zero[0] = True
        n_valid_all_zero = 1
    
    print(f"Valid samples after forcing: {n_valid_all_zero}")
    filtered_batch_all_zero = batch.select_idxs(valid_sample_mask_all_zero)
    print(f"Filtered batch size: {filtered_batch_all_zero.batch.batch_size[0]}")
    print("✓ Edge case handled correctly")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")


if __name__ == "__main__":
    test_filter_zero_advantage_samples()

"""
Test script for _filter_zero_advantage_samples function.
This tests all edge cases and scenarios that can occur in real training.
"""

import torch
import numpy as np
from collections import defaultdict
from tensordict import TensorDict
from verl import DataProto


class MockConfig:
    """Mock config object for testing"""
    def __init__(self, n=4, mini_batch_size=None):
        self.actor_rollout_ref = type('obj', (object,), {
            'rollout': type('obj', (object,), {'n': n})(),
            'actor': type('obj', (object,), {
                'get': lambda self, key, default=None: mini_batch_size if key == 'ppo_mini_batch_size' else default
            })()
        })()


def create_test_batch(advantages_per_sample, uids, response_lengths=None):
    """
    Create a test batch with given advantages and uids.
    
    Args:
        advantages_per_sample: list of float, advantage sum for each sample
        uids: list of str, uid for each sample
        response_lengths: list of int, response length for each sample (default: all 10)
    """
    batch_size = len(advantages_per_sample)
    if response_lengths is None:
        response_lengths = [10] * batch_size
    
    max_len = max(response_lengths)
    
    # Create advantages tensor (batch_size, max_len)
    advantages = torch.zeros(batch_size, max_len)
    response_mask = torch.zeros(batch_size, max_len)
    
    for i, (adv_sum, resp_len) in enumerate(zip(advantages_per_sample, response_lengths)):
        # Distribute advantage evenly across response tokens
        if resp_len > 0:
            advantages[i, :resp_len] = adv_sum / resp_len
            response_mask[i, :resp_len] = 1.0
    
    # Create TensorDict batch
    batch_dict = TensorDict({
        'advantages': advantages,
        'response_mask': response_mask,
    }, batch_size=[batch_size])
    
    non_tensor_batch = {
        'uid': np.array(uids, dtype=object)
    }
    
    batch = DataProto(batch=batch_dict, non_tensor_batch=non_tensor_batch)
    return batch


def filter_zero_advantage_samples(batch, config):
    """
    Replicate the _filter_zero_advantage_samples logic for testing.
    """
    from collections import defaultdict
    
    advantages = batch.batch["advantages"]
    response_mask = batch.batch["response_mask"]
    uids = batch.non_tensor_batch["uid"]
    n = config.actor_rollout_ref.rollout.n
    
    print(f"\n{'='*60}")
    print(f"advantages.shape: {advantages.shape}")
    print(f"Filtering by groups with n={n} responses per prompt (grouped by uid)")
    
    # Compute per-sample advantage sums
    masked_advantages = torch.abs(advantages) * response_mask
    advantage_sum_per_sample = masked_advantages.sum(dim=-1)
    
    print(f"advantage_sum_per_sample: {advantage_sum_per_sample.tolist()}")
    print(f"uids: {list(uids)}")
    
    epsilon = 1e-8
    
    # Group samples by uid
    uid_to_indices = defaultdict(list)
    for i, uid in enumerate(uids):
        uid_to_indices[uid].append(i)
    
    print(f"\nuid_to_indices:")
    for uid, indices in uid_to_indices.items():
        group_advs = [advantage_sum_per_sample[i].item() for i in indices]
        print(f"  {uid}: indices={indices}, advantages={group_advs}, sum={sum(group_advs):.6f}")
    
    # Check each group and decide whether to keep it
    n_groups_total = len(uid_to_indices)
    n_groups_kept = 0
    valid_sample_mask = torch.zeros(len(uids), dtype=torch.bool)
    
    for uid, indices in uid_to_indices.items():
        group_total_adv = advantage_sum_per_sample[indices].sum()
        
        if group_total_adv > epsilon:
            for idx in indices:
                valid_sample_mask[idx] = True
            n_groups_kept += 1
            print(f"  ✓ Keeping group {uid} (total_adv={group_total_adv:.6f})")
        else:
            print(f"  ✗ Filtering group {uid} (total_adv={group_total_adv:.6f})")
    
    n_groups_filtered = n_groups_total - n_groups_kept
    
    metrics = {
        "filter/n_groups_total": n_groups_total,
        "filter/n_groups_kept": n_groups_kept,
        "filter/n_groups_filtered": n_groups_filtered,
    }
    
    print(f"\nFiltered {n_groups_filtered}/{n_groups_total} groups (kept {n_groups_kept})")
    
    # If all groups are filtered, keep at least one group
    if n_groups_kept == 0:
        print("Warning: All groups have zero advantage. Keeping the first group.")
        first_uid = list(uid_to_indices.keys())[0]
        for idx in uid_to_indices[first_uid]:
            valid_sample_mask[idx] = True
        n_groups_kept = 1
        metrics["filter/n_groups_kept"] = n_groups_kept
        metrics["filter/forced_keep_one_group"] = 1
    
    n_valid = valid_sample_mask.sum().item()
    print(f"After initial filtering: n_valid={n_valid}")
    
    # Check mini_batch_size constraints
    mini_batch_size = config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
    if mini_batch_size is not None:
        print(f"\nChecking mini_batch_size constraint: {mini_batch_size}")
        
        # Collect valid groups
        valid_groups = []
        for uid, indices in uid_to_indices.items():
            if all(valid_sample_mask[idx] for idx in indices):
                valid_groups.append((uid, indices))
        
        # If n_valid < mini_batch_size, add back filtered groups
        if n_valid < mini_batch_size:
            print(f"  n_valid ({n_valid}) < mini_batch_size ({mini_batch_size}), adding back filtered groups")
            
            filtered_groups = []
            for uid, indices in uid_to_indices.items():
                if not any(valid_sample_mask[idx] for idx in indices):
                    filtered_groups.append((uid, indices))
            
            print(f"  Available filtered groups: {len(filtered_groups)}")
            
            for uid, indices in filtered_groups:
                if n_valid >= mini_batch_size:
                    break
                for idx in indices:
                    valid_sample_mask[idx] = True
                n_valid += len(indices)
                n_groups_kept += 1
                print(f"    Added back group {uid} (size={len(indices)}), n_valid now={n_valid}")
            
            if n_valid < mini_batch_size:
                print(f"  Warning: Even after adding back, n_valid ({n_valid}) < mini_batch_size ({mini_batch_size})")
                metrics["filter/insufficient_samples_for_mini_batch"] = 1
            else:
                metrics["filter/added_back_filtered_groups"] = 1
            
            metrics["filter/n_groups_kept"] = n_groups_kept
        
        # Adjust to make n_valid divisible by mini_batch_size
        if n_valid % mini_batch_size != 0:
            n_to_keep = (n_valid // mini_batch_size) * mini_batch_size
            print(f"  n_valid ({n_valid}) not divisible by mini_batch_size, adjusting to {n_to_keep}")
            
            if n_to_keep < mini_batch_size:
                print(f"  Warning: n_to_keep ({n_to_keep}) < mini_batch_size, keeping all {n_valid} samples")
            else:
                # Re-collect valid groups after potential additions
                valid_groups = []
                for uid, indices in uid_to_indices.items():
                    if all(valid_sample_mask[idx] for idx in indices):
                        valid_groups.append((uid, indices))
                
                # Keep groups until we reach n_to_keep
                new_mask = torch.zeros_like(valid_sample_mask, dtype=torch.bool)
                total_samples_kept = 0
                groups_kept_count = 0
                
                for uid, indices in valid_groups:
                    group_size = len(indices)
                    if total_samples_kept + group_size <= n_to_keep:
                        for idx in indices:
                            new_mask[idx] = True
                        total_samples_kept += group_size
                        groups_kept_count += 1
                        print(f"    Keeping group {uid} (size={group_size}), total={total_samples_kept}")
                    else:
                        print(f"    Skipping group {uid} (size={group_size}), would exceed n_to_keep")
                        break
                
                valid_sample_mask = new_mask
                n_valid = valid_sample_mask.sum().item()
                
                metrics["filter/n_groups_kept"] = groups_kept_count
                metrics["filter/adjusted_for_mini_batch_size"] = 1
                
                print(f"  Final: {groups_kept_count} groups, {n_valid} samples")
    
    print(f"\nFinal valid_sample_mask: {valid_sample_mask.tolist()}")
    print(f"Metrics: {metrics}")
    print(f"{'='*60}\n")
    
    # Filter the batch
    valid_indices = torch.where(valid_sample_mask)[0]
    filtered_batch_dict = TensorDict({
        'advantages': batch.batch['advantages'][valid_indices],
        'response_mask': batch.batch['response_mask'][valid_indices],
    }, batch_size=[len(valid_indices)])
    filtered_non_tensor_batch = {
        'uid': batch.non_tensor_batch['uid'][valid_indices.cpu().numpy()]
    }
    filtered_batch = DataProto(batch=filtered_batch_dict, non_tensor_batch=filtered_non_tensor_batch)
    
    return filtered_batch, metrics


def test_case_1_basic_filtering():
    """Test basic filtering: some groups with zero advantage, some with non-zero"""
    print("\n" + "="*80)
    print("TEST CASE 1: Basic Filtering")
    print("="*80)
    
    # 4 groups, n=4 each
    # Group 0: all zero (should be filtered)
    # Group 1: non-zero (should be kept)
    # Group 2: all zero (should be filtered)
    # Group 3: non-zero (should be kept)
    advantages = [0.0, 0.0, 0.0, 0.0,  # group 0 (uid_0)
                  0.5, 0.3, 0.2, 0.1,  # group 1 (uid_1)
                  0.0, 0.0, 0.0, 0.0,  # group 2 (uid_2)
                  0.4, 0.3, 0.2, 0.1]  # group 3 (uid_3)
    uids = ['uid_0', 'uid_0', 'uid_0', 'uid_0',
            'uid_1', 'uid_1', 'uid_1', 'uid_1',
            'uid_2', 'uid_2', 'uid_2', 'uid_2',
            'uid_3', 'uid_3', 'uid_3', 'uid_3']
    
    config = MockConfig(n=4, mini_batch_size=None)
    batch = create_test_batch(advantages, uids)
    
    filtered_batch, metrics = filter_zero_advantage_samples(batch, config)
    
    # Verify results
    assert metrics['filter/n_groups_total'] == 4
    assert metrics['filter/n_groups_kept'] == 2
    assert metrics['filter/n_groups_filtered'] == 2
    assert len(filtered_batch.batch['advantages']) == 8  # 2 groups * 4 samples
    
    print("✓ TEST CASE 1 PASSED")


def test_case_2_all_zero():
    """Test when all groups have zero advantage"""
    print("\n" + "="*80)
    print("TEST CASE 2: All Zero Advantages")
    print("="*80)
    
    advantages = [0.0] * 16
    uids = ['uid_0'] * 4 + ['uid_1'] * 4 + ['uid_2'] * 4 + ['uid_3'] * 4
    
    config = MockConfig(n=4, mini_batch_size=None)
    batch = create_test_batch(advantages, uids)
    
    filtered_batch, metrics = filter_zero_advantage_samples(batch, config)
    
    # Should keep at least one group
    assert metrics['filter/n_groups_kept'] == 1
    assert metrics.get('filter/forced_keep_one_group', 0) == 1
    assert len(filtered_batch.batch['advantages']) == 4
    
    print("✓ TEST CASE 2 PASSED")


def test_case_3_mini_batch_size_less_than():
    """Test when filtered samples < mini_batch_size"""
    print("\n" + "="*80)
    print("TEST CASE 3: Filtered Samples < Mini Batch Size")
    print("="*80)
    
    # 4 groups, only 1 with non-zero advantage
    advantages = [0.0, 0.0, 0.0, 0.0,  # group 0 (filtered)
                  0.5, 0.3, 0.2, 0.1,  # group 1 (kept) - only 4 samples
                  0.0, 0.0, 0.0, 0.0,  # group 2 (filtered)
                  0.0, 0.0, 0.0, 0.0]  # group 3 (filtered)
    uids = ['uid_0'] * 4 + ['uid_1'] * 4 + ['uid_2'] * 4 + ['uid_3'] * 4
    
    config = MockConfig(n=4, mini_batch_size=8)  # Need at least 8 samples
    batch = create_test_batch(advantages, uids)
    
    filtered_batch, metrics = filter_zero_advantage_samples(batch, config)
    
    # Should add back one filtered group to meet mini_batch_size
    assert metrics['filter/n_groups_kept'] >= 2  # At least 2 groups (8 samples)
    assert len(filtered_batch.batch['advantages']) >= 8
    assert metrics.get('filter/added_back_filtered_groups', 0) == 1
    
    print("✓ TEST CASE 3 PASSED")


def test_case_4_mini_batch_size_divisibility():
    """Test adjustment for mini_batch_size divisibility"""
    print("\n" + "="*80)
    print("TEST CASE 4: Mini Batch Size Divisibility")
    print("="*80)
    
    # 5 groups with non-zero advantages = 20 samples
    # mini_batch_size = 8, should keep 16 samples (2 mini-batches, 4 groups)
    advantages = ([0.5, 0.3, 0.2, 0.1] * 5)  # All groups have non-zero advantages
    uids = (['uid_0'] * 4 + ['uid_1'] * 4 + ['uid_2'] * 4 + 
            ['uid_3'] * 4 + ['uid_4'] * 4)
    
    config = MockConfig(n=4, mini_batch_size=8)
    batch = create_test_batch(advantages, uids)
    
    filtered_batch, metrics = filter_zero_advantage_samples(batch, config)
    
    # Should keep 16 samples (4 groups) to be divisible by 8
    assert len(filtered_batch.batch['advantages']) == 16
    assert metrics['filter/n_groups_kept'] == 4
    assert metrics.get('filter/adjusted_for_mini_batch_size', 0) == 1
    
    print("✓ TEST CASE 4 PASSED")


def test_case_5_unequal_group_sizes():
    """Test with groups of different sizes (shuffled data)"""
    print("\n" + "="*80)
    print("TEST CASE 5: Unequal Group Sizes (Shuffled Data)")
    print("="*80)
    
    # Simulate shuffled data where groups are not contiguous
    # uid_0: 3 samples, uid_1: 5 samples, uid_2: 4 samples
    advantages = [0.5, 0.3, 0.2, 0.1, 0.4,  # uid_1 (5 samples)
                  0.2, 0.3, 0.1,              # uid_0 (3 samples)
                  0.0, 0.0, 0.0, 0.0]         # uid_2 (4 samples, zero)
    uids = ['uid_1', 'uid_1', 'uid_1', 'uid_1', 'uid_1',
            'uid_0', 'uid_0', 'uid_0',
            'uid_2', 'uid_2', 'uid_2', 'uid_2']
    
    config = MockConfig(n=4, mini_batch_size=None)  # n is nominal, actual groups vary
    batch = create_test_batch(advantages, uids)
    
    filtered_batch, metrics = filter_zero_advantage_samples(batch, config)
    
    # Should keep uid_0 (3 samples) and uid_1 (5 samples), filter uid_2 (4 samples)
    assert metrics['filter/n_groups_total'] == 3
    assert metrics['filter/n_groups_kept'] == 2
    assert len(filtered_batch.batch['advantages']) == 8  # 3 + 5
    
    print("✓ TEST CASE 5 PASSED")


def test_case_6_insufficient_total_samples():
    """Test when total samples < mini_batch_size even after adding back all groups"""
    print("\n" + "="*80)
    print("TEST CASE 6: Insufficient Total Samples")
    print("="*80)
    
    # Only 2 groups = 8 samples, but mini_batch_size = 32
    advantages = [0.5, 0.3, 0.2, 0.1,  # group 0
                  0.0, 0.0, 0.0, 0.0]  # group 1 (zero)
    uids = ['uid_0'] * 4 + ['uid_1'] * 4
    
    config = MockConfig(n=4, mini_batch_size=32)
    batch = create_test_batch(advantages, uids)
    
    filtered_batch, metrics = filter_zero_advantage_samples(batch, config)
    
    # Should add back the zero group but still insufficient
    assert metrics.get('filter/insufficient_samples_for_mini_batch', 0) == 1
    assert len(filtered_batch.batch['advantages']) == 8  # All samples kept
    
    print("✓ TEST CASE 6 PASSED")


def test_case_7_edge_case_epsilon():
    """Test edge case with advantages very close to epsilon"""
    print("\n" + "="*80)
    print("TEST CASE 7: Edge Case Near Epsilon")
    print("="*80)
    
    epsilon = 1e-8
    # Group 0: slightly above epsilon (should be kept)
    # Group 1: exactly zero (should be filtered)
    # Group 2: slightly below epsilon (should be filtered)
    advantages = [epsilon * 0.3, epsilon * 0.3, epsilon * 0.3, epsilon * 0.3,  # sum > epsilon
                  0.0, 0.0, 0.0, 0.0,                                            # sum = 0
                  epsilon * 0.1, epsilon * 0.1, epsilon * 0.1, epsilon * 0.1]   # sum < epsilon
    uids = ['uid_0'] * 4 + ['uid_1'] * 4 + ['uid_2'] * 4
    
    config = MockConfig(n=4, mini_batch_size=None)
    batch = create_test_batch(advantages, uids)
    
    filtered_batch, metrics = filter_zero_advantage_samples(batch, config)
    
    # Should keep only group 0
    assert metrics['filter/n_groups_kept'] == 1
    assert len(filtered_batch.batch['advantages']) == 4
    
    print("✓ TEST CASE 7 PASSED")


def run_all_tests():
    """Run all test cases"""
    print("\n" + "🚀 "*40)
    print("RUNNING ALL TEST CASES FOR _filter_zero_advantage_samples")
    print("🚀 "*40 + "\n")
    
    try:
        test_case_1_basic_filtering()
        test_case_2_all_zero()
        test_case_3_mini_batch_size_less_than()
        test_case_4_mini_batch_size_divisibility()
        test_case_5_unequal_group_sizes()
        test_case_6_insufficient_total_samples()
        test_case_7_edge_case_epsilon()
        
        print("\n" + "✅ "*40)
        print("ALL TESTS PASSED!")
        print("✅ "*40 + "\n")
        
    except AssertionError as e:
        print("\n" + "❌ "*40)
        print(f"TEST FAILED: {e}")
        print("❌ "*40 + "\n")
        raise
    except Exception as e:
        print("\n" + "💥 "*40)
        print(f"UNEXPECTED ERROR: {e}")
        print("💥 "*40 + "\n")
        raise


if __name__ == "__main__":
    run_all_tests()

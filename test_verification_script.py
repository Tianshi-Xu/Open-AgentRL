"""Test the verification script with mock data to ensure it works correctly."""

import torch
import numpy as np
from tensordict import TensorDict
import sys
sys.path.insert(0, '/home/Open-AgentRL-test')

from verify_negative_samples_in_training import verify_negative_samples


class MockDataProto:
    """Mock DataProto for testing."""
    def __init__(self, batch, non_tensor_batch):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
    
    def __len__(self):
        return self.batch["prompts"].shape[0]


def create_test_batch_with_negatives():
    """Create a test batch with negative samples."""
    
    print("\n" + "="*80)
    print("Creating Test Batch with Negative Samples")
    print("="*80)
    
    # Simulate: 2 original prompts, each repeated 3 times (N=3), plus 2 negative samples
    # Total: 2*3 + 2 = 8 samples
    batch_size = 8
    prompt_len = 10
    response_len = 20
    
    # Create batch tensors
    batch = TensorDict({
        "prompts": torch.randint(0, 100, (batch_size, prompt_len)),
        "responses": torch.randint(0, 100, (batch_size, response_len)),
        "response_mask": torch.ones(batch_size, response_len, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, prompt_len + response_len),
        "input_ids": torch.randint(0, 100, (batch_size, prompt_len + response_len)),
        "position_ids": torch.arange(prompt_len + response_len).unsqueeze(0).expand(batch_size, -1),
    }, batch_size=(batch_size,))
    
    # Create rm_scores: 6 original samples with positive scores, 2 negative with -1
    rm_scores = torch.zeros(batch_size, response_len, dtype=torch.float32)
    
    # Original samples (indices 0-5): positive scores
    original_scores = [0.5, 0.6, 0.7, 0.55, 0.65, 0.75]
    for i in range(6):
        rm_scores[i, response_len - 1] = original_scores[i]
    
    # Negative samples (indices 6-7): score=-1
    rm_scores[6, response_len - 1] = -1.0
    rm_scores[7, response_len - 1] = -1.0
    
    batch["rm_scores"] = rm_scores
    
    # Create UIDs: 
    # - First 3 samples (prompt 1, repeated 3 times) share UID1
    # - Next 3 samples (prompt 2, repeated 3 times) share UID2
    # - Negative sample 1 (from prompt 1) inherits UID1
    # - Negative sample 2 (from prompt 2) inherits UID2
    uid1 = "uuid-prompt-1-abc123"
    uid2 = "uuid-prompt-2-def456"
    
    uids = np.array([
        uid1, uid1, uid1,  # Prompt 1 repeated 3 times
        uid2, uid2, uid2,  # Prompt 2 repeated 3 times
        uid1,              # Negative sample from prompt 1
        uid2,              # Negative sample from prompt 2
    ], dtype=object)
    
    # Create negative_samples metadata
    # This is stored per ORIGINAL sample (before expansion)
    # We have 6 original samples, and 2 of them produced negative samples
    negative_samples_metadata = [
        None,  # Sample 0
        [{"score": -1, "error": "Tool call failed"}],  # Sample 1 produced 1 negative
        None,  # Sample 2
        None,  # Sample 3
        [{"score": -1, "error": "Tool execution error"}],  # Sample 4 produced 1 negative
        None,  # Sample 5
        None,  # Negative sample 1 (expanded, no metadata)
        None,  # Negative sample 2 (expanded, no metadata)
    ]
    
    non_tensor_batch = {
        "uid": uids,
        "negative_samples": np.array(negative_samples_metadata, dtype=object),
        "__num_turns__": np.array([1]*batch_size, dtype=np.int32),
    }
    
    print(f"✓ Created batch with {batch_size} samples")
    print(f"  - 6 original samples (2 prompts × 3 repeats)")
    print(f"  - 2 negative samples")
    print(f"  - 2 unique UIDs")
    print(f"  - Scores: {original_scores + [-1.0, -1.0]}")
    
    return MockDataProto(batch, non_tensor_batch)


def test_verification():
    """Test the verification function."""
    
    print("\n" + "="*80)
    print("TESTING NEGATIVE SAMPLES VERIFICATION")
    print("="*80)
    
    # Test 1: Batch with negative samples
    print("\n" + "="*80)
    print("TEST 1: Batch with Negative Samples")
    print("="*80)
    
    batch = create_test_batch_with_negatives()
    results = verify_negative_samples(batch, step=100, verbose=True)
    
    print("\nTest 1 Results:")
    print(f"  has_negative_samples: {results['has_negative_samples']}")
    print(f"  num_negative_samples: {results['num_negative_samples']}")
    print(f"  num_original_samples: {results['num_original_samples']}")
    print(f"  negative_samples_have_correct_uids: {results['negative_samples_have_correct_uids']}")
    print(f"  negative_samples_have_score_minus_one: {results['negative_samples_have_score_minus_one']}")
    print(f"  groups_with_negative_samples: {results['groups_with_negative_samples']}")
    print(f"  total_groups: {results['total_groups']}")
    print(f"  all_checks_passed: {results['all_checks_passed']}")
    
    assert results['has_negative_samples'], "Should detect negative samples"
    assert results['all_checks_passed'], "All checks should pass"
    
    # Test 2: Batch without negative samples
    print("\n" + "="*80)
    print("TEST 2: Batch without Negative Samples")
    print("="*80)
    
    batch_no_neg = MockDataProto(
        batch=TensorDict({
            "prompts": torch.randint(0, 100, (4, 10)),
            "responses": torch.randint(0, 100, (4, 20)),
            "response_mask": torch.ones(4, 20, dtype=torch.long),
            "rm_scores": torch.zeros(4, 20, dtype=torch.float32),
        }, batch_size=(4,)),
        non_tensor_batch={
            "uid": np.array(["uid1", "uid1", "uid2", "uid2"], dtype=object),
            "__num_turns__": np.array([1, 1, 1, 1], dtype=np.int32),
        }
    )
    
    results2 = verify_negative_samples(batch_no_neg, step=101, verbose=True)
    
    print("\nTest 2 Results:")
    print(f"  has_negative_samples: {results2['has_negative_samples']}")
    
    assert not results2['has_negative_samples'], "Should not detect negative samples"
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nThe verification script is working correctly.")
    print("You can now use it in actual training to verify negative samples.")


if __name__ == "__main__":
    test_verification()

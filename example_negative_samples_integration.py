"""
Integration example: Using negative samples with GRPO trainer

This example shows how to integrate the negative samples feature into 
the GRPO training pipeline for agentic reinforcement learning.
"""

import numpy as np
import torch
from verl.protocol import DataProto


def expand_batch_with_negative_samples(batch: DataProto, config) -> DataProto:
    """
    Expand the batch by converting negative samples to regular training samples.
    
    This function:
    1. Extracts negative samples from extra_fields
    2. Converts them to the same format as regular samples
    3. Assigns them the same UID as their parent (for GRPO grouping)
    4. Marks them for special reward handling
    
    Args:
        batch: The original DataProto batch
        config: Configuration object
        
    Returns:
        Expanded DataProto with negative samples as additional training examples
    """
    
    # Check if batch has negative samples
    if "negative_samples" not in batch.non_tensor_batch:
        return batch
    
    negative_samples_array = batch.non_tensor_batch["negative_samples"]
    
    # Count total negative samples
    total_negatives = 0
    for samples in negative_samples_array:
        if samples is not None and isinstance(samples, list):
            total_negatives += len(samples)
    
    if total_negatives == 0:
        return batch
    
    print(f"Found {total_negatives} negative samples to add to batch")
    
    # Get original UIDs for group assignment
    original_uids = batch.non_tensor_batch.get("uid", None)
    if original_uids is None:
        raise ValueError("UID field is required for negative sample processing")
    
    # Get dimensions
    response_length = batch.batch["responses"].shape[1]
    prompt_length = batch.batch["prompts"].shape[1]
    device = batch.batch["responses"].device
    
    # Prepare lists for new samples
    new_responses = []
    new_response_masks = []
    new_rollout_logprobs = []
    new_uids = []
    new_is_negative = []
    
    # Process each sample
    for i, samples in enumerate(negative_samples_array):
        if samples is None or not isinstance(samples, list) or len(samples) == 0:
            continue
        
        parent_uid = original_uids[i]
        
        for neg_sample in samples:
            # Extract data from negative sample
            response_ids = neg_sample.get("response_ids", [])
            response_mask = neg_sample.get("response_mask", [])
            response_logprobs = neg_sample.get("response_logprobs", None)
            
            # Pad or truncate to response_length
            if len(response_ids) < response_length:
                pad_length = response_length - len(response_ids)
                response_ids = response_ids + [0] * pad_length
                response_mask = response_mask + [0] * pad_length
                if response_logprobs is not None:
                    response_logprobs = response_logprobs + [0.0] * pad_length
            
            response_ids = response_ids[:response_length]
            response_mask = response_mask[:response_length]
            if response_logprobs is not None:
                response_logprobs = response_logprobs[:response_length]
            
            # Convert to tensors
            new_responses.append(torch.tensor(response_ids, dtype=torch.long))
            new_response_masks.append(torch.tensor(response_mask, dtype=torch.long))
            
            if response_logprobs is not None:
                new_rollout_logprobs.append(torch.tensor(response_logprobs, dtype=torch.float32))
            
            # Assign same UID as parent (for GRPO grouping)
            new_uids.append(parent_uid)
            new_is_negative.append(True)
    
    # Stack new samples
    if new_responses:
        new_responses_tensor = torch.stack(new_responses).to(device)
        new_response_masks_tensor = torch.stack(new_response_masks).to(device)
        
        # Concatenate with original batch
        batch.batch["responses"] = torch.cat([batch.batch["responses"], new_responses_tensor], dim=0)
        batch.batch["response_mask"] = torch.cat([batch.batch["response_mask"], new_response_masks_tensor], dim=0)
        
        # Handle prompts (reuse parent's prompt for negative samples)
        # This is a simplification - in practice you might want to extract the actual prompt
        # For now, we'll create dummy prompts or reuse
        dummy_prompts = torch.zeros((len(new_responses), prompt_length), dtype=torch.long, device=device)
        batch.batch["prompts"] = torch.cat([batch.batch["prompts"], dummy_prompts], dim=0)
        
        # Handle logprobs if present
        if "rollout_log_probs" in batch.batch and new_rollout_logprobs:
            new_rollout_logprobs_tensor = torch.stack(new_rollout_logprobs).to(device)
            batch.batch["rollout_log_probs"] = torch.cat(
                [batch.batch["rollout_log_probs"], new_rollout_logprobs_tensor], dim=0
            )
        
        # Update UIDs
        new_uids_array = np.array(new_uids, dtype=object)
        batch.non_tensor_batch["uid"] = np.concatenate([original_uids, new_uids_array])
        
        # Add marker for negative samples
        original_is_negative = np.array([False] * len(original_uids))
        new_is_negative_array = np.array(new_is_negative)
        batch.non_tensor_batch["is_negative_sample"] = np.concatenate(
            [original_is_negative, new_is_negative_array]
        )
        
        print(f"Expanded batch from {len(original_uids)} to {len(batch.batch['responses'])} samples")
    
    return batch


def assign_negative_rewards(batch: DataProto, negative_reward: float = -0.5):
    """
    Assign negative rewards to negative samples.
    
    Args:
        batch: DataProto batch with is_negative_sample markers
        negative_reward: Reward value to assign to negative samples
    """
    
    if "is_negative_sample" not in batch.non_tensor_batch:
        return
    
    if "token_level_rewards" not in batch.batch:
        print("Warning: token_level_rewards not found in batch")
        return
    
    is_negative = batch.non_tensor_batch["is_negative_sample"]
    
    # Assign negative rewards to the final token of negative samples
    for i, is_neg in enumerate(is_negative):
        if is_neg:
            # Find the last valid response token
            response_mask = batch.batch["response_mask"][i]
            valid_positions = torch.nonzero(response_mask, as_tuple=True)[0]
            
            if len(valid_positions) > 0:
                last_valid_pos = valid_positions[-1].item()
                batch.batch["token_level_rewards"][i, last_valid_pos] = negative_reward
                print(f"Assigned reward {negative_reward} to negative sample {i} at position {last_valid_pos}")


def log_negative_sample_stats(batch: DataProto):
    """
    Log statistics about negative samples in the batch.
    """
    
    if "negative_samples" not in batch.non_tensor_batch:
        print("No negative samples field found")
        return
    
    negative_samples_array = batch.non_tensor_batch["negative_samples"]
    
    # Count samples
    total_samples = len(negative_samples_array)
    samples_with_negatives = 0
    total_negatives = 0
    
    error_types_count = {}
    
    for samples in negative_samples_array:
        if samples is not None and isinstance(samples, list) and len(samples) > 0:
            samples_with_negatives += 1
            total_negatives += len(samples)
            
            # Count error types
            for neg_sample in samples:
                for error_type in neg_sample.get("error_types", []):
                    error_types_count[error_type] = error_types_count.get(error_type, 0) + 1
    
    print(f"\n{'='*60}")
    print(f"Negative Sample Statistics:")
    print(f"  Total samples in batch: {total_samples}")
    print(f"  Samples with negative examples: {samples_with_negatives} ({samples_with_negatives/total_samples*100:.1f}%)")
    print(f"  Total negative samples: {total_negatives}")
    print(f"  Average negatives per sample: {total_negatives/total_samples:.2f}")
    
    if error_types_count:
        print(f"\n  Error types distribution:")
        for error_type, count in sorted(error_types_count.items(), key=lambda x: x[1], reverse=True):
            print(f"    {error_type}: {count} ({count/total_negatives*100:.1f}%)")
    
    print(f"{'='*60}\n")


# Example usage in trainer
class ExampleTrainerIntegration:
    """
    Example showing how to integrate negative samples in the training loop.
    """
    
    def __init__(self, config):
        self.config = config
        self.enable_negative_samples = config.actor_rollout_ref.rollout.multi_turn.get("save_negative_samples", False)
        self.negative_reward = config.get("negative_sample_reward", -0.5)
    
    def process_rollout_batch(self, batch: DataProto) -> DataProto:
        """
        Process rollout batch, including negative samples if enabled.
        """
        
        if not self.enable_negative_samples:
            return batch
        
        # Log statistics about negative samples
        log_negative_sample_stats(batch)
        
        # Expand batch with negative samples
        batch = expand_batch_with_negative_samples(batch, self.config)
        
        # Assign negative rewards
        assign_negative_rewards(batch, self.negative_reward)
        
        return batch
    
    def fit_step(self, batch: DataProto):
        """
        Single training step with negative sample handling.
        """
        
        # Process negative samples
        batch = self.process_rollout_batch(batch)
        
        # Continue with normal GRPO training
        # The negative samples are now part of their parent's group (same UID)
        # and will receive negative advantages due to their low rewards
        
        # ... rest of training code ...
        
        pass


# Example configuration
example_config = """
# In your YAML config file or command line args:

algorithm:
  adv_estimator: grpo

actor_rollout_ref:
  rollout:
    n: 8  # Generate 8 samples per prompt
    multi_turn:
      enable_tool_rollback: true
      max_tool_retries: 3
      save_negative_samples: true
      max_negative_samples_per_group: 1  # Add at most 1 negative sample per group

# Reward for negative samples (should be lower than normal failures)
negative_sample_reward: -0.5
"""

if __name__ == "__main__":
    print("Negative Samples Integration Example")
    print("=" * 60)
    print("\nThis example demonstrates how to integrate negative samples")
    print("into the GRPO training pipeline.")
    print("\nKey steps:")
    print("1. Extract negative samples from batch.non_tensor_batch")
    print("2. Convert them to regular training samples")
    print("3. Assign same UID as parent (for GRPO grouping)")
    print("4. Assign negative rewards")
    print("5. GRPO will compute advantages across all samples in group")
    print("\nExample configuration:")
    print(example_config)

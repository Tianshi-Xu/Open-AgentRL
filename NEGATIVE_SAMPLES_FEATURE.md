# Negative Samples Feature for Tool Call Rollback

## Overview

This feature allows the system to save failed tool call trajectories as negative samples during the rollback mechanism. These negative samples can be used to teach the model to avoid making similar mistakes, providing a form of negative reinforcement learning.

## Key Features

1. **Failed Trajectory Collection**: When a tool call fails and triggers rollback, the system can save the failed trajectory before attempting to fix it.

2. **Group Size Limit**: To avoid overwhelming the batch with too many negative samples, you can limit the maximum number of negative samples per group (controlled by `max_negative_samples_per_group`).

3. **Complete Sample Information**: Each negative sample includes:
   - Full token IDs (prompt + response)
   - Response mask
   - Log probabilities (if available)
   - Error messages and types
   - Tool call information
   - Turn counts

## Configuration

Add the following parameters to your configuration file:

```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      enable_tool_rollback: true
      max_tool_retries: 3
      save_negative_samples: true  # Enable negative sample saving
      max_negative_samples_per_group: 1  # Max negative samples per group
```

Or via command line:

```bash
actor_rollout_ref.rollout.multi_turn.save_negative_samples=True \
actor_rollout_ref.rollout.multi_turn.max_negative_samples_per_group=1
```

## How It Works

### 1. Rollback Detection
When a tool call error matching the configured error patterns occurs:
- The system creates a checkpoint of the current agent state
- If `save_negative_samples=True` and within the limit, it saves the failed trajectory

### 2. Negative Sample Creation
Each negative sample contains:
```python
{
    "prompt_ids": [...],           # Full prompt token IDs
    "response_ids": [...],         # Response token IDs
    "response_mask": [...],        # Mask for response tokens
    "response_logprobs": [...],    # Log probabilities (if available)
    "error_messages": [...],       # Error messages that triggered rollback
    "error_types": [...],          # Types of errors (e.g., "ImportError")
    "tool_position": "turn_X",     # Position where error occurred
    "assistant_turns": N,          # Number of assistant turns
    "user_turns": M,               # Number of user turns
    "tool_calls": [...]            # Tool call information
}
```

### 3. Integration with Training
The negative samples are added to the `AgentLoopOutput.extra_fields["negative_samples"]` and flow through the data pipeline:

```
AgentLoopOutput 
  └─> extra_fields["negative_samples"] 
      └─> DataProto.non_tensor_batch["negative_samples"]
          └─> Available during advantage computation and training
```

## Processing Negative Samples in Training

The trainer needs to be updated to process these negative samples. Here's how to integrate them:

### Step 1: Extract Negative Samples

In your trainer's data processing pipeline:

```python
# In the generate_sequences or compute_advantage step
if "negative_samples" in batch.non_tensor_batch:
    negative_samples = batch.non_tensor_batch["negative_samples"]
    
    # Each element is an array of negative samples for that original sample
    for i, sample_negatives in enumerate(negative_samples):
        if sample_negatives is not None and len(sample_negatives) > 0:
            # Process each negative sample
            for neg_sample in sample_negatives:
                # neg_sample is a dictionary with the structure shown above
                process_negative_sample(neg_sample)
```

### Step 2: Create Additional Training Samples

You can expand your batch to include the negative samples:

```python
def expand_batch_with_negatives(batch: DataProto, uid_array: np.ndarray) -> DataProto:
    """Expand batch with negative samples as separate training examples."""
    
    if "negative_samples" not in batch.non_tensor_batch:
        return batch
    
    negative_samples_array = batch.non_tensor_batch["negative_samples"]
    original_batch_size = len(batch)
    
    # Collect all negative samples and their corresponding UIDs
    neg_samples_to_add = []
    neg_uids = []
    
    for i, sample_negatives in enumerate(negative_samples_array):
        if sample_negatives is None or len(sample_negatives) == 0:
            continue
            
        base_uid = uid_array[i]
        for j, neg_sample in enumerate(sample_negatives):
            neg_samples_to_add.append(neg_sample)
            # Create a unique UID for negative sample (same group as parent)
            neg_uids.append(base_uid)
    
    if not neg_samples_to_add:
        return batch
    
    # Convert negative samples to tensor format
    neg_batches = []
    for neg_sample in neg_samples_to_add:
        # Pad to match response_length
        response_length = batch.batch["responses"].shape[1]
        response_ids = neg_sample["response_ids"]
        response_mask = neg_sample["response_mask"]
        response_logprobs = neg_sample.get("response_logprobs")
        
        # Pad if needed
        if len(response_ids) < response_length:
            pad_length = response_length - len(response_ids)
            response_ids = response_ids + [0] * pad_length
            response_mask = response_mask + [0] * pad_length
            if response_logprobs is not None:
                response_logprobs = response_logprobs + [0.0] * pad_length
        
        # Truncate if needed
        response_ids = response_ids[:response_length]
        response_mask = response_mask[:response_length]
        if response_logprobs is not None:
            response_logprobs = response_logprobs[:response_length]
        
        # Create tensors
        neg_batch = {
            "responses": torch.tensor([response_ids], dtype=torch.long),
            "response_mask": torch.tensor([response_mask], dtype=torch.long),
        }
        
        if response_logprobs is not None:
            neg_batch["rollout_log_probs"] = torch.tensor([response_logprobs], dtype=torch.float32)
        
        neg_batches.append(neg_batch)
    
    # Concatenate with original batch
    if neg_batches:
        for key in batch.batch.keys():
            if key in neg_batches[0]:
                neg_tensors = [nb[key] for nb in neg_batches]
                batch.batch[key] = torch.cat([batch.batch[key]] + neg_tensors, dim=0)
        
        # Update UIDs
        new_uid_array = np.concatenate([uid_array, np.array(neg_uids)])
        batch.non_tensor_batch["uid"] = new_uid_array
    
    return batch
```

### Step 3: Assign Negative Rewards

Negative samples should receive lower rewards to provide negative reinforcement:

```python
def assign_negative_rewards(batch: DataProto, negative_reward: float = -1.0):
    """Assign negative rewards to negative samples."""
    
    if "token_level_rewards" not in batch.batch:
        return
    
    # Mark which samples are negative (after expansion)
    # This information should be tracked during expansion
    # For simplicity, you can add a marker in non_tensor_batch
    
    if "is_negative_sample" in batch.non_tensor_batch:
        is_negative = batch.non_tensor_batch["is_negative_sample"]
        for i, is_neg in enumerate(is_negative):
            if is_neg:
                # Set the final reward to negative_reward
                response_length = batch.batch["response_mask"][i].sum().item()
                if response_length > 0:
                    batch.batch["token_level_rewards"][i, response_length - 1] = negative_reward
```

## Usage with GRPO

The negative samples work seamlessly with GRPO because:

1. **Group Membership**: Negative samples share the same UID as their parent sample, so they belong to the same group for advantage computation.

2. **Advantage Computation**: In GRPO, advantage is computed as `(reward - group_mean) / group_std`. Negative samples with lower rewards will naturally get negative advantages.

3. **Limit per Group**: The `max_negative_samples_per_group` parameter ensures you don't add too many negative samples to a group, which could skew the group statistics.

## Example Configuration

```yaml
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
```

With this configuration:
- For each prompt, you generate 8 samples
- If tool call errors occur, up to 1 failed trajectory per group is saved
- The final batch may have up to 9 samples per group (8 successful + 1 failed)
- GRPO computes advantages across all 9 samples in the group

## Benefits

1. **Learning from Mistakes**: The model learns to avoid tool call patterns that lead to errors.

2. **Balanced Training**: By limiting negative samples per group, you maintain a good balance between positive and negative examples.

3. **Complete Context**: Each negative sample includes full trajectory information, allowing the model to understand the context of the error.

4. **Compatible with Existing Pipeline**: The feature integrates seamlessly with the existing training pipeline through the extra_fields mechanism.

## Notes

- Negative samples are collected **before** the rollback mechanism attempts to fix the error
- The successful retry (if any) is kept in the main trajectory
- This provides both positive examples (successful trajectories) and negative examples (failed attempts)
- The limit per group prevents overwhelming the batch with too many negative samples

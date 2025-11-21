# Negative Samples Feature - Complete Implementation Guide

## Quick Start

### Enable the Feature

Add to your training configuration:

```bash
actor_rollout_ref.rollout.multi_turn.enable_tool_rollback=True \
actor_rollout_ref.rollout.multi_turn.max_tool_retries=3 \
actor_rollout_ref.rollout.multi_turn.save_negative_samples=True \
actor_rollout_ref.rollout.multi_turn.max_negative_samples_per_group=1
```

### What It Does

1. **Saves Failed Trajectories**: When tool calls fail, the system saves the failed trajectory before attempting to fix it
2. **Creates Negative Samples**: Failed trajectories become training samples with negative rewards
3. **Maintains Group Structure**: Negative samples share the same UID as their parent for GRPO grouping
4. **Limits Per Group**: Prevents overwhelming batches with too many negative samples

## Implementation Details

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Tool Call Execution                                         │
├─────────────────────────────────────────────────────────────┤
│ 1. Execute tool call                                        │
│ 2. Check for errors matching rollback_on_errors patterns    │
│ 3. If error detected and save_negative_samples=True:        │
│    ├─ Check: negative_samples_count < max_per_group?        │
│    ├─ If yes: Create negative sample                        │
│    └─ Save to agent_data.negative_samples[]                 │
│ 4. Create checkpoint of current state                       │
│ 5. Append error feedback to messages                        │
│ 6. Regenerate tool call (rollback)                          │
│ 7. If regeneration succeeds: Continue with new tool call    │
│ 8. If max retries exceeded: Notify model and continue       │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Failed Tool Call
      ↓
Rollback Detection (matches error patterns)
      ↓
Save Negative Sample (if enabled & within limit)
      ├─ Extract current state:
      │  ├─ prompt_ids
      │  ├─ response_ids
      │  ├─ response_mask
      │  ├─ response_logprobs
      │  ├─ error_messages
      │  ├─ error_types
      │  ├─ tool_position
      │  └─ tool_calls info
      ↓
Store in AgentData.negative_samples[]
      ↓
Create Checkpoint & Rollback
      ↓
Regenerate Tool Call
      ↓
On Success: Continue with fixed trajectory
      ↓
At End of Rollout:
      ├─ Add negative_samples to AgentLoopOutput.extra_fields
      ↓
DataProto.non_tensor_batch["negative_samples"]
      ↓
Trainer Processing:
      ├─ Extract negative samples
      ├─ Convert to training samples
      ├─ Assign same UID as parent (for GRPO grouping)
      ├─ Assign negative rewards
      ↓
GRPO Advantage Computation:
      ├─ Group by UID
      ├─ Compute group mean & std
      ├─ Calculate advantage = (reward - mean) / std
      └─ Negative samples get negative advantages
```

## Configuration Reference

### All Parameters

```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      # Enable rollback mechanism
      enable_tool_rollback: true
      
      # Maximum retries per tool call position
      max_tool_retries: 3
      
      # Save failed trajectories as negative samples
      save_negative_samples: true
      
      # Maximum negative samples per group
      # This limits how many negative samples can be added to each group
      # to prevent overwhelming the batch
      max_negative_samples_per_group: 1
      
      # Error patterns that trigger rollback and negative sample creation
      rollback_on_errors:
        - "ImportError"
        - "ModuleNotFoundError"
        - "SyntaxError"
        - "IndentationError"
        - "NameError"
        - "TypeError"
        - "IndexError"
        - "worker_timeout"
```

### Parameter Interactions

| enable_tool_rollback | save_negative_samples | Result |
|---------------------|----------------------|---------|
| False | False | No rollback, no negative samples |
| False | True | No rollback, no negative samples (rollback required) |
| True | False | Rollback enabled, no negative samples saved |
| True | True | Rollback enabled, negative samples saved |

## Trainer Integration

### Method 1: Process in Rollout Pipeline

```python
# In your trainer's generate_sequences or post-rollout processing:

def process_rollout_batch(self, batch: DataProto) -> DataProto:
    """Process batch including negative samples."""
    
    if not self.config.actor_rollout_ref.rollout.multi_turn.get("save_negative_samples", False):
        return batch
    
    # Extract and log negative sample statistics
    if "negative_samples" in batch.non_tensor_batch:
        log_negative_sample_stats(batch)
    
    return batch
```

### Method 2: Expand Batch Before Training

```python
def expand_batch_with_negatives(batch: DataProto, config) -> DataProto:
    """
    Expand batch by converting negative samples to training samples.
    
    Key steps:
    1. Extract negative samples from non_tensor_batch
    2. Convert to tensor format (pad/truncate to match dimensions)
    3. Assign same UID as parent for GRPO grouping
    4. Mark as negative samples for reward assignment
    5. Concatenate with original batch
    """
    
    if "negative_samples" not in batch.non_tensor_batch:
        return batch
    
    negative_samples_array = batch.non_tensor_batch["negative_samples"]
    original_uids = batch.non_tensor_batch["uid"]
    
    # Collect samples to add
    new_samples = []
    new_uids = []
    
    for i, neg_samples in enumerate(negative_samples_array):
        if neg_samples is None or len(neg_samples) == 0:
            continue
        
        parent_uid = original_uids[i]
        
        for neg_sample in neg_samples:
            # Process and add negative sample
            processed_sample = process_negative_sample(
                neg_sample, 
                batch.batch["responses"].shape[1]  # response_length
            )
            new_samples.append(processed_sample)
            new_uids.append(parent_uid)  # Same UID for grouping
    
    if new_samples:
        # Concatenate with original batch
        batch = concatenate_samples(batch, new_samples, new_uids)
    
    return batch
```

### Method 3: Handle in Reward Computation

```python
def assign_negative_rewards(batch: DataProto, negative_reward: float = -0.5):
    """Assign negative rewards to negative samples."""
    
    if "is_negative_sample" not in batch.non_tensor_batch:
        return
    
    is_negative = batch.non_tensor_batch["is_negative_sample"]
    
    for i, is_neg in enumerate(is_negative):
        if is_neg:
            # Assign negative reward at final valid token
            response_mask = batch.batch["response_mask"][i]
            valid_positions = torch.nonzero(response_mask, as_tuple=True)[0]
            if len(valid_positions) > 0:
                last_pos = valid_positions[-1].item()
                batch.batch["token_level_rewards"][i, last_pos] = negative_reward
```

## GRPO Integration

### How It Works with GRPO

GRPO computes advantages as: `advantage = (reward - group_mean) / group_std`

**Example with 8 successful + 1 failed sample per group:**

```
Group 123:
  Sample 1: reward = 1.0 (successful)
  Sample 2: reward = 0.8 (successful)
  Sample 3: reward = 0.9 (successful)
  Sample 4: reward = 1.0 (successful)
  Sample 5: reward = 0.7 (successful)
  Sample 6: reward = 0.9 (successful)
  Sample 7: reward = 0.8 (successful)
  Sample 8: reward = 1.0 (successful)
  Sample 9: reward = -0.5 (negative sample)

Group mean = (1.0 + 0.8 + 0.9 + 1.0 + 0.7 + 0.9 + 0.8 + 1.0 - 0.5) / 9 = 0.733
Group std = 0.469

Advantages:
  Sample 1: (1.0 - 0.733) / 0.469 = +0.57
  Sample 2: (0.8 - 0.733) / 0.469 = +0.14
  ...
  Sample 9: (-0.5 - 0.733) / 0.469 = -2.63 (strong negative)
```

The negative sample gets a strong negative advantage, teaching the model to avoid such trajectories.

## Best Practices

### 1. Choose Appropriate Group Limits

```yaml
# For small groups (n=4-8): Use 1 negative sample
max_negative_samples_per_group: 1

# For medium groups (n=8-16): Use 1-2 negative samples
max_negative_samples_per_group: 2

# For large groups (n>16): Use 2-3 negative samples
max_negative_samples_per_group: 3
```

**Rule of thumb**: Keep negative samples < 20% of group size

### 2. Tune Negative Rewards

```python
# Start conservative
negative_reward = -0.5  # Half of typical positive reward

# For critical errors, use stronger negative signal
if "SyntaxError" in error_types:
    negative_reward = -1.0

# For minor errors, use weaker signal
if "ImportError" in error_types:
    negative_reward = -0.3
```

### 3. Monitor Negative Sample Statistics

```python
def log_negative_sample_stats(batch: DataProto):
    """Log statistics about negative samples."""
    
    if "negative_samples" not in batch.non_tensor_batch:
        return
    
    negative_samples_array = batch.non_tensor_batch["negative_samples"]
    
    total_samples = len(negative_samples_array)
    samples_with_negatives = 0
    total_negatives = 0
    error_counts = defaultdict(int)
    
    for samples in negative_samples_array:
        if samples and len(samples) > 0:
            samples_with_negatives += 1
            total_negatives += len(samples)
            
            for neg_sample in samples:
                for error_type in neg_sample.get("error_types", []):
                    error_counts[error_type] += 1
    
    print(f"Negative Sample Stats:")
    print(f"  Samples with negatives: {samples_with_negatives}/{total_samples}")
    print(f"  Total negative samples: {total_negatives}")
    print(f"  Error distribution: {dict(error_counts)}")
```

### 4. Validate Log Probabilities

Ensure negative samples have valid log probabilities for gradient computation:

```python
def validate_logprobs(negative_sample: dict):
    """Validate that negative sample has valid logprobs."""
    
    if "response_logprobs" not in negative_sample:
        logger.warning("Negative sample missing logprobs")
        return False
    
    logprobs = negative_sample["response_logprobs"]
    
    if logprobs is None or len(logprobs) == 0:
        logger.warning("Negative sample has empty logprobs")
        return False
    
    # Check for NaN or Inf
    if any(np.isnan(lp) or np.isinf(lp) for lp in logprobs):
        logger.warning("Negative sample has invalid logprobs")
        return False
    
    return True
```

## Troubleshooting

### Issue 1: No Negative Samples Generated

**Symptoms**: `negative_samples` field is always empty

**Possible causes**:
1. `save_negative_samples=False` in config
2. No tool call errors occur
3. Errors don't match `rollback_on_errors` patterns
4. `max_negative_samples_per_group` limit already reached

**Solutions**:
```bash
# Check configuration
actor_rollout_ref.rollout.multi_turn.save_negative_samples=True

# Add more error patterns
actor_rollout_ref.rollout.multi_turn.rollback_on_errors="['ImportError','SyntaxError','TypeError','ValueError']"

# Increase limit
actor_rollout_ref.rollout.multi_turn.max_negative_samples_per_group=2
```

### Issue 2: Too Many Negative Samples

**Symptoms**: Batch size explodes, training becomes slow

**Solutions**:
```bash
# Reduce limit per group
actor_rollout_ref.rollout.multi_turn.max_negative_samples_per_group=1

# Make error patterns more selective
actor_rollout_ref.rollout.multi_turn.rollback_on_errors="['ImportError','SyntaxError']"
```

### Issue 3: Negative Samples Not Affecting Training

**Symptoms**: Model doesn't learn from negative samples

**Possible causes**:
1. Negative rewards not assigned correctly
2. Negative samples not properly grouped with parent
3. Negative reward value too small

**Solutions**:
```python
# Increase negative reward magnitude
negative_reward = -1.0

# Verify UID assignment
assert neg_sample_uid == parent_uid

# Verify reward assignment
assert batch.batch["token_level_rewards"][negative_sample_idx].min() < 0
```

## Testing

### Unit Tests

Run the provided unit tests:

```bash
python test_negative_samples.py
```

### Validation Script

Run the validation script:

```bash
python validate_negative_samples.py
```

### Integration Test

Test with a small training run:

```bash
python -m recipe.demystify.custom_main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=4 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.multi_turn.enable_tool_rollback=True \
    actor_rollout_ref.rollout.multi_turn.save_negative_samples=True \
    actor_rollout_ref.rollout.multi_turn.max_negative_samples_per_group=1 \
    trainer.total_epochs=1
```

## Performance Considerations

### Memory Impact

- Each negative sample adds ~1KB to `extra_fields`
- With `max_negative_samples_per_group=1` and `n=8`:
  - Original: 8 samples per prompt
  - With negatives: Up to 9 samples per prompt
  - Memory increase: ~12.5%

### Computation Impact

- Negligible overhead during rollout (< 1%)
- GRPO advantage computation scales linearly with group size
- With negatives, group size increases by ~12.5% (1 negative per 8 samples)

### Recommendations

For optimal performance:

```yaml
# Balance quality and efficiency
actor_rollout_ref:
  rollout:
    n: 8  # Base samples per prompt
    multi_turn:
      max_negative_samples_per_group: 1  # Add 1 negative = 12.5% increase
```

## Examples

### Minimal Example

```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      enable_tool_rollback: true
      save_negative_samples: true
```

### Production Example

```yaml
algorithm:
  adv_estimator: grpo

actor_rollout_ref:
  rollout:
    n: 8
    multi_turn:
      enable_tool_rollback: true
      max_tool_retries: 3
      save_negative_samples: true
      max_negative_samples_per_group: 1
      rollback_on_errors:
        - "ImportError"
        - "ModuleNotFoundError"
        - "SyntaxError"

trainer:
  negative_sample_reward: -0.5
```

## References

- Implementation: `verl/verl/experimental/agent_loop/tool_agent_loop.py`
- Configuration: `verl/verl/workers/config/rollout.py`
- Tests: `test_negative_samples.py`
- Documentation: `NEGATIVE_SAMPLES_FEATURE.md`
- Integration: `example_negative_samples_integration.py`

#!/usr/bin/env python3
"""
Real-world simulation test for checkpoint mechanism.
This test simulates the actual behavior during rollback.
"""
import copy
from collections import defaultdict


class MockAgentData:
    """Mock AgentData for testing."""
    def __init__(self):
        self.prompt_ids = [1, 2, 3, 4, 5]
        self.response_ids = [6, 7, 8]
        self.response_mask = [1, 1, 1]
        self.response_logprobs = [0.1, 0.2, 0.3]
        self.messages = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "code"}
        ]
        self.image_data = None
        self.assistant_turns = 1
        self.user_turns = 1
        self.tool_retry_counts = defaultdict(int)
        self.max_tool_retries = 3


def create_checkpoint_original(agent_data):
    """Original implementation with deep copy."""
    return {
        "prompt_ids": copy.deepcopy(agent_data.prompt_ids),
        "response_ids": copy.deepcopy(agent_data.response_ids),
        "response_mask": copy.deepcopy(agent_data.response_mask),
        "response_logprobs": copy.deepcopy(agent_data.response_logprobs) if agent_data.response_logprobs else None,
        "messages": copy.deepcopy(agent_data.messages),
        "image_data": agent_data.image_data,
        "assistant_turns": agent_data.assistant_turns,
        "user_turns": agent_data.user_turns,
    }


def create_checkpoint_optimized(agent_data):
    """Optimized implementation with minimal copying."""
    return {
        "prompt_ids": list(agent_data.prompt_ids),
        "response_ids": agent_data.response_ids,  # Direct reference
        "response_mask": list(agent_data.response_mask),
        "response_logprobs": list(agent_data.response_logprobs) if agent_data.response_logprobs else None,
        "messages": copy.deepcopy(agent_data.messages),
        "image_data": agent_data.image_data,
        "assistant_turns": agent_data.assistant_turns,
        "user_turns": agent_data.user_turns,
    }


def restore_checkpoint(agent_data, checkpoint):
    """Restore checkpoint."""
    agent_data.prompt_ids = checkpoint["prompt_ids"]
    agent_data.response_ids = checkpoint["response_ids"]
    agent_data.response_mask = checkpoint["response_mask"]
    agent_data.response_logprobs = checkpoint["response_logprobs"]
    agent_data.messages = checkpoint["messages"]
    agent_data.image_data = checkpoint["image_data"]
    agent_data.assistant_turns = checkpoint["assistant_turns"]
    agent_data.user_turns = checkpoint["user_turns"]


def simulate_error_feedback_phase(agent_data):
    """Simulate: append error feedback + encode + update states."""
    # Step 1: Append error message
    error_message = {"role": "user", "content": "Error: SyntaxError"}
    agent_data.messages.append(error_message)
    
    # Step 2: Encode error prompt (simulate tokenization)
    error_prompt_ids = [100, 101, 102]  # Simulated token ids
    
    # Step 3: Update states (THIS IS THE KEY PART)
    agent_data.prompt_ids += error_prompt_ids
    agent_data.response_mask += [0] * len(error_prompt_ids)
    if agent_data.response_logprobs:
        agent_data.response_logprobs += [0.0] * len(error_prompt_ids)


def simulate_llm_regeneration(agent_data):
    """Simulate: LLM generates new response."""
    # Step 1: assistant_turns increment
    agent_data.assistant_turns += 1
    
    # Step 2: Get new response
    new_response_ids = [200, 201, 202, 203]
    agent_data.response_ids = new_response_ids
    
    # Step 3: Update states (THIS IS THE KEY PART)
    agent_data.prompt_ids += agent_data.response_ids
    agent_data.response_mask += [1] * len(agent_data.response_ids)
    if agent_data.response_logprobs:
        agent_data.response_logprobs += [0.5, 0.6, 0.7, 0.8]


def test_checkpoint_mechanism(checkpoint_func, name):
    """Test checkpoint mechanism with real workflow."""
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"{'='*80}")
    
    # Initial state
    agent_data = MockAgentData()
    print(f"\n1. Initial state:")
    print(f"   prompt_ids: {agent_data.prompt_ids}")
    print(f"   response_ids: {agent_data.response_ids}")
    print(f"   response_mask: {agent_data.response_mask}")
    print(f"   response_logprobs: {agent_data.response_logprobs}")
    print(f"   messages: {agent_data.messages}")
    print(f"   assistant_turns: {agent_data.assistant_turns}")
    
    # Create checkpoint BEFORE tool execution
    checkpoint = checkpoint_func(agent_data)
    print(f"\n2. Checkpoint created (before tool execution)")
    
    # Tool execution fails, now we enter error handling
    print(f"\n3. Tool execution failed, entering error feedback phase...")
    simulate_error_feedback_phase(agent_data)
    print(f"   After error feedback:")
    print(f"   prompt_ids: {agent_data.prompt_ids}")
    print(f"   response_mask: {agent_data.response_mask}")
    print(f"   response_logprobs: {agent_data.response_logprobs}")
    print(f"   messages: {agent_data.messages}")
    
    # LLM regenerates
    print(f"\n4. LLM regenerating new tool call...")
    simulate_llm_regeneration(agent_data)
    print(f"   After LLM regeneration:")
    print(f"   prompt_ids: {agent_data.prompt_ids}")
    print(f"   response_ids: {agent_data.response_ids}")
    print(f"   response_mask: {agent_data.response_mask}")
    print(f"   response_logprobs: {agent_data.response_logprobs}")
    print(f"   assistant_turns: {agent_data.assistant_turns}")
    
    # Restore checkpoint
    print(f"\n5. Restoring checkpoint...")
    restore_checkpoint(agent_data, checkpoint)
    print(f"   After restore:")
    print(f"   prompt_ids: {agent_data.prompt_ids}")
    print(f"   response_ids: {agent_data.response_ids}")
    print(f"   response_mask: {agent_data.response_mask}")
    print(f"   response_logprobs: {agent_data.response_logprobs}")
    print(f"   messages: {agent_data.messages}")
    print(f"   assistant_turns: {agent_data.assistant_turns}")
    
    # Verify correctness
    print(f"\n6. Verification:")
    expected_prompt_ids = [1, 2, 3, 4, 5]
    expected_response_mask = [1, 1, 1]
    expected_response_logprobs = [0.1, 0.2, 0.3]
    expected_messages_count = 2
    expected_assistant_turns = 1
    
    checks = [
        (agent_data.prompt_ids == expected_prompt_ids, 
         f"prompt_ids: {agent_data.prompt_ids} == {expected_prompt_ids}"),
        (agent_data.response_mask == expected_response_mask, 
         f"response_mask: {agent_data.response_mask} == {expected_response_mask}"),
        (agent_data.response_logprobs == expected_response_logprobs, 
         f"response_logprobs: {agent_data.response_logprobs} == {expected_response_logprobs}"),
        (len(agent_data.messages) == expected_messages_count, 
         f"messages count: {len(agent_data.messages)} == {expected_messages_count}"),
        (agent_data.assistant_turns == expected_assistant_turns, 
         f"assistant_turns: {agent_data.assistant_turns} == {expected_assistant_turns}"),
    ]
    
    all_passed = True
    for passed, msg in checks:
        status = "✓" if passed else "✗"
        print(f"   {status} {msg}")
        if not passed:
            all_passed = False
    
    return all_passed


def test_mutability_issue():
    """Test if shallow copy causes mutability issues."""
    print(f"\n{'='*80}")
    print(f"Testing: Mutability Issues with Shallow Copy")
    print(f"{'='*80}")
    
    agent_data = MockAgentData()
    
    # Create checkpoint with shallow copy
    checkpoint = {
        "prompt_ids": list(agent_data.prompt_ids),
        "response_mask": list(agent_data.response_mask),
    }
    
    print(f"\n1. Original prompt_ids: {agent_data.prompt_ids}")
    print(f"   Checkpoint prompt_ids: {checkpoint['prompt_ids']}")
    print(f"   Are they the same object? {agent_data.prompt_ids is checkpoint['prompt_ids']}")
    
    # Modify agent_data
    agent_data.prompt_ids += [100, 101, 102]
    
    print(f"\n2. After modifying agent_data.prompt_ids += [100, 101, 102]:")
    print(f"   agent_data.prompt_ids: {agent_data.prompt_ids}")
    print(f"   checkpoint prompt_ids: {checkpoint['prompt_ids']}")
    
    if checkpoint['prompt_ids'] == [1, 2, 3, 4, 5]:
        print(f"   ✓ Checkpoint is NOT affected (shallow copy is safe)")
        return True
    else:
        print(f"   ✗ Checkpoint IS affected (shallow copy is UNSAFE)")
        return False


if __name__ == "__main__":
    print("="*80)
    print("Real-world Checkpoint Mechanism Test")
    print("="*80)
    
    # Test mutability first
    mutability_ok = test_mutability_issue()
    
    # Test original implementation
    original_ok = test_checkpoint_mechanism(create_checkpoint_original, "Original (Deep Copy)")
    
    # Test optimized implementation
    optimized_ok = test_checkpoint_mechanism(create_checkpoint_optimized, "Optimized (Minimal Copy)")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Mutability test: {'✓ PASSED' if mutability_ok else '✗ FAILED'}")
    print(f"Original implementation: {'✓ PASSED' if original_ok else '✗ FAILED'}")
    print(f"Optimized implementation: {'✓ PASSED' if optimized_ok else '✗ FAILED'}")
    
    if mutability_ok and original_ok and optimized_ok:
        print(f"\n✓✓✓ ALL TESTS PASSED - Both implementations are correct!")
        print(f"\nRecommendation: Use optimized version for better performance")
        exit(0)
    else:
        print(f"\n✗✗✗ SOME TESTS FAILED - Need to fix before production!")
        exit(1)

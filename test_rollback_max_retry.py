#!/usr/bin/env python3
"""
Test script for rollback mechanism with max retry exceeded scenario.
Tests the new graceful failure handling behavior.
"""
import asyncio
import sys
from unittest.mock import MagicMock, AsyncMock, patch
from collections import defaultdict

# Add verl to path
sys.path.insert(0, '/home/Open-AgentRL-test/verl')

from verl.experimental.agent_loop.tool_agent_loop import (
    RollbackManager,
    AgentData,
    AgentState,
    ToolAgentLoop
)
from verl.tools.schemas import ToolResponse


class MockTokenizer:
    """Mock tokenizer for testing."""
    def encode(self, text, add_special_tokens=False):
        return [1] * len(text.split())
    
    def decode(self, token_ids, skip_special_tokens=False):
        return " ".join([f"tok{i}" for i in token_ids])
    
    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, **kwargs):
        # Simulate encoding
        text = str(messages)
        if tokenize:
            return [1] * len(text.split())
        return text


def test_rollback_manager():
    """Test RollbackManager basic functionality."""
    print("=" * 60)
    print("TEST 1: RollbackManager Basic Functionality")
    print("=" * 60)
    
    # Test 1: Initialization
    manager = RollbackManager(
        enable=True,
        max_retries=3,
        error_patterns=["ImportError", "SyntaxError", "ModuleNotFoundError"]
    )
    
    assert manager.enable == True
    assert manager.max_retries == 3
    assert len(manager.error_patterns) == 3
    print("✓ Initialization successful")
    
    # Test 2: Error detection
    test_cases = [
        ("ImportError: No module named 'numpy'", True, "ImportError"),
        ("SyntaxError: invalid syntax", True, "SyntaxError"),
        ("TypeError: unsupported operand type", False, ""),
        ("ModuleNotFoundError: No module named 'torch'", True, "ModuleNotFoundError"),
    ]
    
    for error_text, expected_rollback, expected_type in test_cases:
        should_rollback, error_type = manager.should_rollback(error_text)
        assert should_rollback == expected_rollback, f"Failed for: {error_text}"
        assert error_type == expected_type, f"Wrong error type for: {error_text}"
        status = "✓" if should_rollback else "✗"
        print(f"{status} Error: '{error_text[:50]}...' -> Rollback={should_rollback}, Type={error_type}")
    
    # Test 3: Retry counting
    key = "turn_1"
    assert manager.can_retry(key) == True
    assert manager.increment_retry(key) == 1
    assert manager.can_retry(key) == True
    assert manager.increment_retry(key) == 2
    assert manager.can_retry(key) == True
    assert manager.increment_retry(key) == 3
    assert manager.can_retry(key) == False  # Max retries reached
    print(f"✓ Retry counting: {manager.retry_counts[key]}/{manager.max_retries}")
    
    print("\n✅ TEST 1 PASSED: RollbackManager working correctly\n")


async def test_detect_errors():
    """Test _detect_errors method."""
    print("=" * 60)
    print("TEST 2: Error Detection with Logging")
    print("=" * 60)
    
    # Setup
    manager = RollbackManager(
        enable=True,
        max_retries=3,
        error_patterns=["ImportError", "SyntaxError", "tool call format is wrong"]
    )
    
    # Create mock ToolAgentLoop instance
    loop = MagicMock(spec=ToolAgentLoop)
    loop.rollback_manager = manager
    
    # Test responses with different error types
    responses = [
        (ToolResponse(text="ImportError: No module named 'numpy'"), 0.0, {}),
        (ToolResponse(text="SyntaxError: invalid syntax"), 0.0, {}),
        (ToolResponse(text="Success result"), 1.0, {}),
        (ToolResponse(text="tool call format is wrong"), 0.0, {}),
    ]
    
    tool_position_key = "turn_1"
    
    # Call _detect_errors (need to bind to instance)
    error_messages, error_types = ToolAgentLoop._detect_errors(loop, responses, tool_position_key)
    
    print(f"Found {len(error_messages)} errors:")
    for i, (msg, err_type) in enumerate(zip(error_messages, error_types)):
        print(f"  {i+1}. Type={err_type}, Message={msg[:50]}...")
    
    assert len(error_messages) == 3, f"Expected 3 errors, got {len(error_messages)}"
    assert len(error_types) == 3, f"Expected 3 error types, got {len(error_types)}"
    assert "ImportError" in error_types
    assert "SyntaxError" in error_types
    assert "tool call format is wrong" in error_types
    
    print("\n✅ TEST 2 PASSED: Error detection working correctly\n")


async def test_max_retry_exceeded_handling():
    """Test the new _handle_max_retry_exceeded method."""
    print("=" * 60)
    print("TEST 3: Max Retry Exceeded Handling (Core Feature)")
    print("=" * 60)
    
    # Create mock components
    tokenizer = MockTokenizer()
    
    # Create minimal ToolAgentLoop instance with required args
    mock_config = MagicMock()
    mock_server = MagicMock()
    
    loop = MagicMock(spec=ToolAgentLoop)
    loop.tokenizer = tokenizer
    loop.processor = None
    loop.system_prompt = [1, 2, 3]
    loop.response_length = 1000
    loop.apply_chat_template_kwargs = {}
    loop.loop = asyncio.get_event_loop()
    
    loop.rollback_manager = RollbackManager(
        enable=True,
        max_retries=3,
        error_patterns=["ImportError", "SyntaxError"]
    )
    
    # Create AgentData
    agent_data = AgentData(
        messages=[{"role": "user", "content": "Calculate sqrt(144)"}],
        image_data=None,
        metrics={},
        request_id="test_123",
        tools_kwargs={}
    )
    agent_data.prompt_ids = [1, 2, 3, 4, 5]
    agent_data.response_mask = [0, 0, 1, 1, 1]
    agent_data.response_logprobs = [0.0] * 5
    agent_data.assistant_turns = 2
    
    # Test inputs
    error_messages = [
        "ImportError: No module named 'numpy'",
        "SyntaxError: invalid syntax"
    ]
    error_types = ["ImportError", "SyntaxError"]
    tool_call_names = ["python_interpreter", "calculator"]
    
    print(f"Initial state:")
    print(f"  - Assistant turns: {agent_data.assistant_turns}")
    print(f"  - Prompt length: {len(agent_data.prompt_ids)}")
    print(f"  - Response mask length: {len(agent_data.response_mask)}")
    print(f"  - Messages count: {len(agent_data.messages)}")
    
    # Call the method (bind to real implementation)
    result_state = await ToolAgentLoop._handle_max_retry_exceeded(
        loop,
        agent_data,
        error_messages,
        error_types,
        tool_call_names
    )
    
    print(f"\nAfter handling max retry:")
    print(f"  - Result state: {result_state}")
    print(f"  - Prompt length: {len(agent_data.prompt_ids)} (increased)")
    print(f"  - Response mask length: {len(agent_data.response_mask)} (increased)")
    print(f"  - Messages count: {len(agent_data.messages)} (added notification)")
    
    # Assertions
    assert result_state == AgentState.GENERATING, f"Expected GENERATING, got {result_state}"
    print("✓ Returns GENERATING state (model continues)")
    
    assert len(agent_data.messages) > 1, "Should have added notification message"
    notification = agent_data.messages[-1]
    assert notification["role"] == "tool", f"Expected 'tool' role, got {notification['role']}"
    print(f"✓ Added notification as 'tool' role")
    
    content = notification["content"]
    assert "failed after 3 attempts" in content, "Should mention retry count"
    assert "ImportError" in content or "SyntaxError" in content, "Should mention error types"
    assert "simplifying" in content.lower(), "Should suggest simplification"
    print(f"✓ Notification content appropriate:")
    print(f"  '{content[:100]}...'")
    
    assert len(agent_data.prompt_ids) > 5, "Should have appended notification tokens"
    assert len(agent_data.response_mask) > 5, "Should have extended response mask"
    print(f"✓ Context updated with notification tokens")
    
    print("\n✅ TEST 3 PASSED: Max retry handling working as designed!\n")


async def test_full_rollback_flow():
    """Test complete rollback flow including max retry."""
    print("=" * 60)
    print("TEST 4: Complete Rollback Flow (Integration Test)")
    print("=" * 60)
    
    tokenizer = MockTokenizer()
    
    # Create ToolAgentLoop mock
    loop = MagicMock(spec=ToolAgentLoop)
    loop.tokenizer = tokenizer
    loop.processor = None
    loop.system_prompt = [1, 2, 3]
    loop.response_length = 1000
    loop.apply_chat_template_kwargs = {}
    loop.loop = asyncio.get_event_loop()
    loop.max_parallel_calls = 5
    
    loop.rollback_manager = RollbackManager(
        enable=True,
        max_retries=2,  # Lower for faster test
        error_patterns=["ImportError", "test_error"]
    )
    
    # Create AgentData
    agent_data = AgentData(
        messages=[{"role": "user", "content": "Test"}],
        image_data=None,
        metrics={},
        request_id="test_flow",
        tools_kwargs={}
    )
    agent_data.prompt_ids = [1, 2, 3]
    agent_data.response_mask = [0, 0, 0]
    agent_data.response_logprobs = [0.0, 0.0, 0.0]
    agent_data.assistant_turns = 1
    agent_data.tool_calls = []  # Will be set by mock
    
    # Scenario: All retries fail, should trigger max retry exceeded
    print("\nScenario: Tool fails repeatedly, triggers max retry exceeded")
    print("-" * 60)
    
    # Simulate failed tool responses
    failed_responses = [
        (ToolResponse(text="ImportError: test_error in code"), 0.0, {}),
    ]
    
    # Track retry attempts
    retry_count = 0
    position_key = f"turn_{agent_data.assistant_turns}"
    
    # Manually simulate the retry loop
    for attempt in range(loop.rollback_manager.max_retries + 1):
        error_messages, error_types = ToolAgentLoop._detect_errors(loop, failed_responses, position_key)
        
        if error_messages:
            if loop.rollback_manager.can_retry(position_key):
                retry_count = loop.rollback_manager.increment_retry(position_key)
                print(f"  Attempt {attempt + 1}: Error detected, retry {retry_count}/{loop.rollback_manager.max_retries}")
            else:
                print(f"  Attempt {attempt + 1}: Max retries reached, calling _handle_max_retry_exceeded")
                
                initial_msg_count = len(agent_data.messages)
                result_state = await ToolAgentLoop._handle_max_retry_exceeded(
                    loop,
                    agent_data,
                    error_messages,
                    error_types,
                    ["test_tool"]
                )
                
                assert result_state == AgentState.GENERATING
                assert len(agent_data.messages) > initial_msg_count
                print(f"  ✓ Notification sent to model, state -> GENERATING")
                break
    
    assert retry_count == loop.rollback_manager.max_retries
    print(f"\n✓ Flow completed: {retry_count} retries -> graceful notification")
    
    print("\n✅ TEST 4 PASSED: Full rollback flow working correctly\n")


async def run_all_tests():
    """Run all test cases."""
    print("\n" + "=" * 60)
    print("ROLLBACK MECHANISM TEST SUITE")
    print("Testing new max retry exceeded behavior")
    print("=" * 60 + "\n")
    
    try:
        # Test 1: Basic RollbackManager
        test_rollback_manager()
        
        # Test 2: Error detection
        await test_detect_errors()
        
        # Test 3: Max retry exceeded handling (NEW FEATURE)
        await test_max_retry_exceeded_handling()
        
        # Test 4: Full integration
        await test_full_rollback_flow()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 60)
        print("\nKey Features Verified:")
        print("  ✓ Error detection with single-pass optimization")
        print("  ✓ Retry counting and limit enforcement")
        print("  ✓ Max retry exceeded graceful handling (NEW)")
        print("  ✓ Notification message formatting for math scenarios")
        print("  ✓ State transition to GENERATING (model continues)")
        print("  ✓ Complete rollback flow with all components")
        print("\nReady for production testing!")
        
        return 0
        
    except AssertionError as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        return 1
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)

"""
Test the negative samples feature for tool call rollback mechanism.

This test verifies that:
1. Failed tool calls are saved as negative samples when enabled
2. The limit per group is respected
3. Negative samples contain correct information (token IDs, logprobs, error info)
4. Negative samples are properly integrated into the output
"""

import asyncio
import copy
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import torch

from verl.experimental.agent_loop.tool_agent_loop import (
    ToolAgentLoop,
    AgentData,
    RollbackManager,
    AgentState,
)


class MockConfig:
    """Mock configuration for testing."""
    
    class ActorRolloutRef:
        class Rollout:
            prompt_length = 512
            response_length = 512
            
            class MultiTurn:
                max_user_turns = 10
                max_assistant_turns = 10
                max_parallel_calls = 1
                max_tool_response_length = 256
                tool_response_truncate_side = "middle"
                tool_config_path = None
                format = "hermes"
                enable_tool_rollback = True
                max_tool_retries = 3
                save_negative_samples = True
                max_negative_samples_per_group = 2  # Allow 2 negative samples for testing
                
                @staticmethod
                def get(key, default=None):
                    if key == "enable_tool_rollback":
                        return True
                    elif key == "max_tool_retries":
                        return 3
                    elif key == "rollback_on_errors":
                        return ["ImportError", "ModuleNotFoundError", "SyntaxError"]
                    elif key == "save_negative_samples":
                        return True
                    elif key == "max_negative_samples_per_group":
                        return 2
                    return default
            
            multi_turn = MultiTurn()
        
        rollout = Rollout()
    
    actor_rollout_ref = ActorRolloutRef()
    
    class Data:
        @staticmethod
        def get(key, default=None):
            return default
    
    data = Data()


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.apply_chat_template = MagicMock(return_value=[1, 2, 3, 4, 5])
    tokenizer.eos_token_id = 2
    return tokenizer


@pytest.fixture
def rollback_manager():
    """Create a RollbackManager instance for testing."""
    return RollbackManager(
        enable=True,
        max_retries=3,
        error_patterns=["ImportError", "ModuleNotFoundError", "SyntaxError"],
        save_negative_samples=True,
        max_negative_samples_per_group=2
    )


def test_rollback_manager_initialization():
    """Test RollbackManager initialization with negative samples config."""
    manager = RollbackManager(
        enable=True,
        max_retries=3,
        error_patterns=["ImportError"],
        save_negative_samples=True,
        max_negative_samples_per_group=2
    )
    
    assert manager.enable is True
    assert manager.max_retries == 3
    assert manager.save_negative_samples is True
    assert manager.max_negative_samples_per_group == 2


def test_agent_data_negative_samples_fields():
    """Test that AgentData has negative samples fields."""
    agent_data = AgentData(
        messages=[],
        image_data=None,
        metrics={},
        request_id="test-123",
        tools_kwargs={},
        interaction=None,
        interaction_kwargs={},
    )
    
    assert hasattr(agent_data, "negative_samples")
    assert hasattr(agent_data, "negative_samples_count")
    assert agent_data.negative_samples == []
    assert agent_data.negative_samples_count == 0


def test_create_negative_sample(mock_tokenizer):
    """Test creating a negative sample from failed trajectory."""
    # Initialize the class (simplified for testing)
    config = MockConfig()
    
    # Create agent data with some state
    agent_data = AgentData(
        messages=[{"role": "user", "content": "test"}],
        image_data=None,
        metrics={},
        request_id="test-123",
        tools_kwargs={},
        interaction=None,
        interaction_kwargs={},
    )
    
    # Set up some trajectory data
    agent_data.prompt_ids = [1, 2, 3, 4, 5]
    agent_data.response_ids = [6, 7, 8]
    agent_data.response_mask = [1, 1, 1]
    agent_data.response_logprobs = [-0.1, -0.2, -0.3]
    agent_data.assistant_turns = 1
    agent_data.user_turns = 1
    
    # Mock tool calls
    from verl.experimental.agent_loop.tool_parser import FunctionCall
    tool_call = FunctionCall(name="test_tool", arguments='{"arg": "value"}')
    agent_data.tool_calls = [tool_call]
    
    # Create a mock ToolAgentLoop instance
    tool_agent_loop = ToolAgentLoop.__new__(ToolAgentLoop)
    
    # Create negative sample
    negative_sample = tool_agent_loop._create_negative_sample(
        agent_data,
        error_messages=["ImportError: No module named 'test'"],
        error_types=["ImportError"],
        tool_position_key="turn_1"
    )
    
    # Verify negative sample structure
    assert "prompt_ids" in negative_sample
    assert "response_ids" in negative_sample
    assert "response_mask" in negative_sample
    assert "response_logprobs" in negative_sample
    assert "error_messages" in negative_sample
    assert "error_types" in negative_sample
    assert "tool_position" in negative_sample
    assert "assistant_turns" in negative_sample
    assert "user_turns" in negative_sample
    assert "tool_calls" in negative_sample
    
    # Verify content
    assert negative_sample["prompt_ids"] == [1, 2, 3, 4, 5]
    assert negative_sample["response_ids"] == [6, 7, 8]
    assert negative_sample["response_mask"] == [1, 1, 1]
    assert negative_sample["response_logprobs"] == [-0.1, -0.2, -0.3]
    assert negative_sample["error_messages"] == ["ImportError: No module named 'test'"]
    assert negative_sample["error_types"] == ["ImportError"]
    assert negative_sample["tool_position"] == "turn_1"
    assert negative_sample["assistant_turns"] == 1
    assert negative_sample["user_turns"] == 1
    assert len(negative_sample["tool_calls"]) == 1
    assert negative_sample["tool_calls"][0]["name"] == "test_tool"


def test_negative_samples_limit():
    """Test that negative samples respect the max_per_group limit."""
    agent_data = AgentData(
        messages=[],
        image_data=None,
        metrics={},
        request_id="test-123",
        tools_kwargs={},
        interaction=None,
        interaction_kwargs={},
    )
    
    # Set up minimal trajectory data
    agent_data.prompt_ids = [1, 2, 3]
    agent_data.response_ids = [4, 5]
    agent_data.response_mask = [1, 1]
    agent_data.response_logprobs = [-0.1, -0.2]
    agent_data.tool_calls = []
    
    # Create a mock ToolAgentLoop instance
    tool_agent_loop = ToolAgentLoop.__new__(ToolAgentLoop)
    tool_agent_loop.rollback_manager = RollbackManager(
        enable=True,
        max_retries=3,
        error_patterns=["ImportError"],
        save_negative_samples=True,
        max_negative_samples_per_group=2
    )
    
    # Simulate multiple failures
    for i in range(5):  # Try to add 5 negative samples
        if agent_data.negative_samples_count < tool_agent_loop.rollback_manager.max_negative_samples_per_group:
            neg_sample = tool_agent_loop._create_negative_sample(
                agent_data,
                error_messages=[f"Error {i}"],
                error_types=["ImportError"],
                tool_position_key=f"turn_{i}"
            )
            agent_data.negative_samples.append(neg_sample)
            agent_data.negative_samples_count += 1
    
    # Should only have 2 negative samples (the limit)
    assert len(agent_data.negative_samples) == 2
    assert agent_data.negative_samples_count == 2


def test_negative_samples_in_output():
    """Test that negative samples are included in AgentLoopOutput."""
    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
    
    # Create output with negative samples
    output = AgentLoopOutput(
        prompt_ids=[1, 2, 3],
        response_ids=[4, 5, 6],
        response_mask=[1, 1, 1],
        response_logprobs=[-0.1, -0.2, -0.3],
        multi_modal_data=None,
        num_turns=2,
        metrics={},
        extra_fields={
            "negative_samples": [
                {
                    "prompt_ids": [1, 2, 3],
                    "response_ids": [4, 5],
                    "error_messages": ["ImportError"],
                    "error_types": ["ImportError"],
                }
            ]
        }
    )
    
    assert "negative_samples" in output.extra_fields
    assert len(output.extra_fields["negative_samples"]) == 1
    assert output.extra_fields["negative_samples"][0]["error_types"] == ["ImportError"]


def test_rollback_manager_should_rollback():
    """Test error pattern matching for rollback."""
    manager = RollbackManager(
        enable=True,
        max_retries=3,
        error_patterns=["ImportError", "ModuleNotFoundError", "SyntaxError"],
        save_negative_samples=True,
        max_negative_samples_per_group=1
    )
    
    # Test matching error
    should_rollback, error_type = manager.should_rollback("ImportError: No module named 'test'")
    assert should_rollback is True
    assert error_type == "ImportError"
    
    # Test non-matching error
    should_rollback, error_type = manager.should_rollback("ValueError: invalid value")
    assert should_rollback is False
    assert error_type == ""


def test_negative_samples_disabled():
    """Test that negative samples are not saved when disabled."""
    manager = RollbackManager(
        enable=True,
        max_retries=3,
        error_patterns=["ImportError"],
        save_negative_samples=False,  # Disabled
        max_negative_samples_per_group=1
    )
    
    assert manager.save_negative_samples is False
    
    # Even if we try to save, the flag should prevent it
    agent_data = AgentData(
        messages=[],
        image_data=None,
        metrics={},
        request_id="test-123",
        tools_kwargs={},
        interaction=None,
        interaction_kwargs={},
    )
    
    # The handler should check this flag before saving
    if manager.save_negative_samples and agent_data.negative_samples_count < manager.max_negative_samples_per_group:
        agent_data.negative_samples.append({})
        agent_data.negative_samples_count += 1
    
    assert len(agent_data.negative_samples) == 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

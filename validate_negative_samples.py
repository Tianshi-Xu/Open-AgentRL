#!/usr/bin/env python3
"""
Validation script for negative samples feature.

This script performs basic validation that the implementation is working correctly:
1. Configuration parameters are recognized
2. Negative samples are created correctly
3. Data structures are compatible with existing code
"""

import sys


def validate_configuration():
    """Validate that configuration parameters are recognized."""
    print("=" * 70)
    print("Testing configuration parameters...")
    print("=" * 70)
    
    try:
        from verl.workers.config.rollout import MultiTurnConfig
        
        # Create config with negative sample parameters
        config = MultiTurnConfig(
            save_negative_samples=True,
            max_negative_samples_per_group=2
        )
        
        assert hasattr(config, 'save_negative_samples'), "Missing save_negative_samples attribute"
        assert hasattr(config, 'max_negative_samples_per_group'), "Missing max_negative_samples_per_group attribute"
        assert config.save_negative_samples == True, "save_negative_samples not set correctly"
        assert config.max_negative_samples_per_group == 2, "max_negative_samples_per_group not set correctly"
        
        print("✓ Configuration parameters validated successfully")
        print(f"  - save_negative_samples: {config.save_negative_samples}")
        print(f"  - max_negative_samples_per_group: {config.max_negative_samples_per_group}")
        return True
        
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        return False


def validate_rollback_manager():
    """Validate RollbackManager with negative sample support."""
    print("\n" + "=" * 70)
    print("Testing RollbackManager...")
    print("=" * 70)
    
    try:
        from verl.experimental.agent_loop.tool_agent_loop import RollbackManager
        
        # Create manager with negative sample support
        manager = RollbackManager(
            enable=True,
            max_retries=3,
            error_patterns=["ImportError", "SyntaxError"],
            save_negative_samples=True,
            max_negative_samples_per_group=2
        )
        
        assert manager.save_negative_samples == True, "save_negative_samples not set"
        assert manager.max_negative_samples_per_group == 2, "max_negative_samples_per_group not set"
        
        # Test error detection
        should_rollback, error_type = manager.should_rollback("ImportError: No module")
        assert should_rollback == True, "Error detection failed"
        assert error_type == "ImportError", f"Wrong error type: {error_type}"
        
        print("✓ RollbackManager validated successfully")
        print(f"  - save_negative_samples: {manager.save_negative_samples}")
        print(f"  - max_negative_samples_per_group: {manager.max_negative_samples_per_group}")
        print(f"  - Error detection working: {error_type}")
        return True
        
    except Exception as e:
        print(f"✗ RollbackManager validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_agent_data():
    """Validate AgentData has negative sample fields."""
    print("\n" + "=" * 70)
    print("Testing AgentData...")
    print("=" * 70)
    
    try:
        from verl.experimental.agent_loop.tool_agent_loop import AgentData
        
        # Create agent data
        agent_data = AgentData(
            messages=[],
            image_data=None,
            metrics={},
            request_id="test-123",
            tools_kwargs={},
            interaction=None,
            interaction_kwargs={},
        )
        
        assert hasattr(agent_data, 'negative_samples'), "Missing negative_samples field"
        assert hasattr(agent_data, 'negative_samples_count'), "Missing negative_samples_count field"
        assert agent_data.negative_samples == [], "negative_samples not initialized correctly"
        assert agent_data.negative_samples_count == 0, "negative_samples_count not initialized correctly"
        
        print("✓ AgentData validated successfully")
        print(f"  - Has negative_samples field: {hasattr(agent_data, 'negative_samples')}")
        print(f"  - Has negative_samples_count field: {hasattr(agent_data, 'negative_samples_count')}")
        print(f"  - Initial count: {agent_data.negative_samples_count}")
        return True
        
    except Exception as e:
        print(f"✗ AgentData validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_negative_sample_creation():
    """Validate negative sample creation."""
    print("\n" + "=" * 70)
    print("Testing negative sample creation...")
    print("=" * 70)
    
    try:
        from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop, AgentData
        from verl.experimental.agent_loop.tool_parser import FunctionCall
        
        # Create a tool agent loop instance
        tool_agent_loop = ToolAgentLoop.__new__(ToolAgentLoop)
        
        # Create agent data with sample trajectory
        agent_data = AgentData(
            messages=[{"role": "user", "content": "test"}],
            image_data=None,
            metrics={},
            request_id="test-123",
            tools_kwargs={},
            interaction=None,
            interaction_kwargs={},
        )
        
        agent_data.prompt_ids = [1, 2, 3, 4, 5]
        agent_data.response_ids = [6, 7, 8]
        agent_data.response_mask = [1, 1, 1]
        agent_data.response_logprobs = [-0.1, -0.2, -0.3]
        agent_data.assistant_turns = 1
        agent_data.user_turns = 1
        agent_data.tool_calls = [FunctionCall(name="test_tool", arguments='{"key": "value"}')]
        
        # Create negative sample
        neg_sample = tool_agent_loop._create_negative_sample(
            agent_data,
            error_messages=["ImportError: test"],
            error_types=["ImportError"],
            tool_position_key="turn_1"
        )
        
        # Validate structure
        required_keys = [
            "prompt_ids", "response_ids", "response_mask", "response_logprobs",
            "error_messages", "error_types", "tool_position",
            "assistant_turns", "user_turns", "tool_calls"
        ]
        
        for key in required_keys:
            assert key in neg_sample, f"Missing key: {key}"
        
        # Validate content
        assert neg_sample["prompt_ids"] == [1, 2, 3, 4, 5], "prompt_ids mismatch"
        assert neg_sample["response_ids"] == [6, 7, 8], "response_ids mismatch"
        assert neg_sample["error_types"] == ["ImportError"], "error_types mismatch"
        assert len(neg_sample["tool_calls"]) == 1, "tool_calls count mismatch"
        
        print("✓ Negative sample creation validated successfully")
        print(f"  - All required fields present: {len(required_keys)}")
        print(f"  - Sample structure correct")
        print(f"  - Error info preserved: {neg_sample['error_types']}")
        return True
        
    except Exception as e:
        print(f"✗ Negative sample creation validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_agent_loop_output():
    """Validate that AgentLoopOutput can contain negative samples."""
    print("\n" + "=" * 70)
    print("Testing AgentLoopOutput with negative samples...")
    print("=" * 70)
    
    try:
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
        
        assert "negative_samples" in output.extra_fields, "negative_samples not in extra_fields"
        assert len(output.extra_fields["negative_samples"]) == 1, "Wrong number of negative samples"
        
        print("✓ AgentLoopOutput validated successfully")
        print(f"  - Can contain negative_samples in extra_fields: True")
        print(f"  - Sample count: {len(output.extra_fields['negative_samples'])}")
        return True
        
    except Exception as e:
        print(f"✗ AgentLoopOutput validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("\n")
    print("*" * 70)
    print("NEGATIVE SAMPLES FEATURE VALIDATION")
    print("*" * 70)
    
    results = []
    
    # Run all validation tests
    results.append(("Configuration", validate_configuration()))
    results.append(("RollbackManager", validate_rollback_manager()))
    results.append(("AgentData", validate_agent_data()))
    results.append(("Negative Sample Creation", validate_negative_sample_creation()))
    results.append(("AgentLoopOutput", validate_agent_loop_output()))
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n🎉 All validations passed! The implementation is working correctly.")
        return 0
    else:
        print("\n❌ Some validations failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

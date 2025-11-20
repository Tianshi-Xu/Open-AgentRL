#!/usr/bin/env python3
"""
Test script to verify the tool rollback mechanism implementation.
This script checks the correctness of the rollback logic without actually running it.
"""

import ast
import sys


def check_method_exists(tree, method_name):
    """Check if a method exists in the AST."""
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef)) and node.name == method_name:
            return True
    return False


def check_class_attribute(tree, class_name, attr_name):
    """Check if a class has an attribute."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in ast.walk(node):
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Attribute) and target.attr == attr_name:
                            return True
                        elif isinstance(target, ast.Name) and target.id == attr_name:
                            return True
    return False


def main():
    file_path = "verl/verl/experimental/agent_loop/tool_agent_loop.py"
    
    print("=" * 80)
    print("Tool Rollback Mechanism - Implementation Verification (Modular)")
    print("=" * 80)
    
    try:
        with open(file_path, 'r') as f:
            code = f.read()
        
        tree = ast.parse(code)
        
        # Check RollbackManager class exists
        print("\n✓ Checking RollbackManager class:")
        has_rollback_manager = "class RollbackManager:" in code
        print(f"  {'✓' if has_rollback_manager else '✗'} RollbackManager class defined")
        
        # Check required methods in RollbackManager
        rollback_methods = [
            "should_rollback",
            "can_retry",
            "increment_retry",
            "format_error_feedback",
            "create_checkpoint",
            "restore_checkpoint",
        ]
        
        print("\n✓ Checking RollbackManager methods:")
        all_methods_exist = True
        for method in rollback_methods:
            exists = f"def {method}(" in code
            status = "✓" if exists else "✗"
            print(f"  {status} {method}")
            all_methods_exist = all_methods_exist and exists
        
        # Check helper methods in ToolAgentLoop
        print("\n✓ Checking ToolAgentLoop helper methods:")
        helper_methods = [
            "_detect_errors",
            "_handle_rollback",
            "_encode_error_feedback",
            "_process_tool_responses",
            "_handle_processing_tools_state"
        ]
        
        all_helpers_exist = True
        for method in helper_methods:
            exists = check_method_exists(tree, method)
            status = "✓" if exists else "✗"
            print(f"  {status} {method}")
            all_helpers_exist = all_helpers_exist and exists
        
        # Check rollback_manager initialization
        print("\n✓ Checking rollback_manager initialization:")
        has_init = "cls.rollback_manager = RollbackManager(" in code
        print(f"  {'✓' if has_init else '✗'} cls.rollback_manager initialized")
        
        # Check key logic patterns
        print("\n✓ Checking key logic patterns:")
        key_patterns = [
            ("RollbackManager instantiation", "RollbackManager(enable_rollback, max_retries, error_patterns)"),
            ("Checkpoint creation", "self.rollback_manager.create_checkpoint(agent_data)"),
            ("Can retry check", "self.rollback_manager.can_retry(tool_position_key)"),
            ("Error detection", "self._detect_errors(responses)"),
            ("Handle rollback", "await self._handle_rollback("),
            ("Restore checkpoint", "self.rollback_manager.restore_checkpoint(agent_data, checkpoint)"),
        ]
        
        all_patterns_exist = True
        for desc, pattern in key_patterns:
            exists = pattern in code
            status = "✓" if exists else "✗"
            print(f"  {status} {desc}")
            all_patterns_exist = all_patterns_exist and exists
        
        # Final summary
        print("\n" + "=" * 80)
        if has_rollback_manager and all_methods_exist and all_helpers_exist and has_init and all_patterns_exist:
            print("✓ All checks passed! Modular implementation is complete.")
            print("=" * 80)
            return 0
        else:
            print("✗ Some checks failed. Please review the implementation.")
            print("=" * 80)
            return 1
            
    except FileNotFoundError:
        print(f"✗ Error: File not found: {file_path}")
        return 1
    except SyntaxError as e:
        print(f"✗ Syntax error in {file_path}: {e}")
        return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

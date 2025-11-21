# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import copy
import json
import json_repair
import logging
import os
import time
from collections import defaultdict
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import add_generation_prompt_for_gpt_oss, format_gpt_oss_tool_response_manually
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Global tool call statistics
_tool_stats = {
    "total": 0,
    "success": 0,
    "failed": 0,
    "rollback_triggered": 0,
    "rollback_recovered": 0,
    "rollback_failed": 0
}
_tool_stats_lock = asyncio.Lock()
_LOG_EVERY_N_CALLS = 20  # Log every 20 tool calls


async def _record_tool_stat(success: bool):
    """Record tool call statistics and log periodically."""
    async with _tool_stats_lock:
        _tool_stats["total"] += 1
        _tool_stats["success" if success else "failed"] += 1
        
        total = _tool_stats["total"]
        if total % _LOG_EVERY_N_CALLS == 0:
            success_count = _tool_stats["success"]
            failed_count = _tool_stats["failed"]
            success_rate = success_count / total if total > 0 else 0.0
            
            rollback_triggered = _tool_stats["rollback_triggered"]
            rollback_recovered = _tool_stats["rollback_recovered"]
            rollback_failed = _tool_stats["rollback_failed"]
            rollback_pending = rollback_triggered - rollback_recovered - rollback_failed
            
            # Always show rollback stats to make it clear whether rollback is working
            ### DEBUG ###
            if rollback_triggered > 0:
                rollback_recovery_rate = rollback_recovered / rollback_triggered
                logger.warning(
                    f"📊 Tool Stats - Total: {total}, Success: {success_count} ({success_rate:.2%}), "
                    f"Failed: {failed_count} | Rollback: {rollback_triggered} triggered, "
                    f"{rollback_recovered} recovered ({rollback_recovery_rate:.2%}), {rollback_failed} failed, "
                    f"{rollback_pending} pending"
                )
            else:
                # Show that rollback stats are all 0 (either disabled or no rollback errors detected)
                logger.warning(
                    f"📊 Tool Stats - Total: {total}, Success: {success_count} ({success_rate:.2%}), "
                    f"Failed: {failed_count} | Rollback: 0 triggered (no rollback errors or disabled)"
                )
            ### DEBUG ###


async def _record_rollback_trigger():
    """Record that a rollback was triggered (first failure of a turn)."""
    async with _tool_stats_lock:
        _tool_stats["rollback_triggered"] += 1


async def _record_rollback_recovery():
    """Record that a rollback successfully recovered (turn succeeded after retries)."""
    async with _tool_stats_lock:
        _tool_stats["rollback_recovered"] += 1


async def _record_rollback_final_failure():
    """Record that a rollback ultimately failed (max retry exceeded)."""
    async with _tool_stats_lock:
        _tool_stats["rollback_failed"] += 1


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class RollbackManager:
    """Manages rollback mechanism for tool call errors."""
    
    def __init__(self, enable: bool, max_retries: int, error_patterns: list[str], 
                 save_negative_samples: bool = False, max_negative_samples_per_group: int = 1):
        self.enable = enable
        self.max_retries = max_retries
        self.error_patterns = error_patterns
        self.save_negative_samples = save_negative_samples
        self.max_negative_samples_per_group = max_negative_samples_per_group
        
    def should_rollback(self, error_text: str) -> tuple[bool, str]:
        """Check if error should trigger rollback.
        
        Returns:
            tuple[bool, str]: (should_rollback, error_type)
        """
        if not self.enable:
            return False, ""
        for pattern in self.error_patterns:
            if pattern.lower() in error_text.lower():
                # print(f"error pattern:{pattern}, error_text:{error_text}")
                return True, pattern
        return False, ""
    
    def can_retry(self, retry_counts: dict[str, int], position_key: str) -> bool:
        """Check if retry is allowed at this position."""
        return retry_counts[position_key] < self.max_retries
    
    def increment_retry(self, retry_counts: dict[str, int], position_key: str) -> int:
        """Increment retry count and return new count."""
        retry_counts[position_key] += 1
        return retry_counts[position_key]
    
    def format_error_feedback(self, error_messages: list[str]) -> str:
        """Format error feedback for LLM."""
        feedback = "The previous tool call(s) failed with the following error(s):\n"
        for i, error in enumerate(error_messages, 1):
            feedback += f"{i}. {error}\n"
        feedback += "\nPlease correct the error and generate a new tool call."
        return feedback
    
    def create_checkpoint(self, agent_data: "AgentData") -> dict[str, Any]:
        """Create a checkpoint of current agent state."""
        return {
            "prompt_ids": list(agent_data.prompt_ids),
            "response_ids": agent_data.response_ids,
            "response_mask": list(agent_data.response_mask),
            "response_logprobs": list(agent_data.response_logprobs) if agent_data.response_logprobs else None,
            "messages": copy.deepcopy(agent_data.messages),
            "image_data": agent_data.image_data,
            "assistant_turns": agent_data.assistant_turns,
            "user_turns": agent_data.user_turns,
        }
    
    def restore_checkpoint(self, agent_data: "AgentData", checkpoint: dict[str, Any]):
        """Restore agent state from checkpoint."""
        agent_data.prompt_ids = checkpoint["prompt_ids"]
        agent_data.response_ids = checkpoint["response_ids"]
        agent_data.response_mask = checkpoint["response_mask"]
        agent_data.response_logprobs = checkpoint["response_logprobs"]
        agent_data.messages = checkpoint["messages"]
        agent_data.image_data = checkpoint["image_data"]
        agent_data.assistant_turns = checkpoint["assistant_turns"]
        agent_data.user_turns = checkpoint["user_turns"]


class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: Any,
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []
        
        # Safety mechanisms to prevent infinite loops
        self.total_tool_attempts = 0  # Track total tool call attempts across all turns
        self.disable_rollback_after_max_retry = False  # Flag to disable rollback after max retry exceeded
        self.rollback_recovered_turns: set[str] = set()  # Track turns that have successfully recovered
        self.retry_counts: dict[str, int] = defaultdict(int)  # Track retry counts for each turn
        
        # Negative samples for failed tool calls
        self.negative_samples: list[dict[str, Any]] = []  # Store failed trajectories as negative samples
        self.negative_samples_count = 0  # Track number of negative samples collected


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        cls.tool_parser_name = config.actor_rollout_ref.rollout.multi_turn.format
        print(f"Initialized tools: {cls.tools}")

        # Initialize rollback manager
        enable_rollback = config.actor_rollout_ref.rollout.multi_turn.get("enable_tool_rollback", False)
        max_retries = config.actor_rollout_ref.rollout.multi_turn.get("max_tool_retries", 3)
        error_patterns = config.actor_rollout_ref.rollout.multi_turn.get(
            "rollback_on_errors",
            ["ImportError", "ModuleNotFoundError", "SyntaxError", "IndentationError", "NameError"],
        )
        save_negative_samples = config.actor_rollout_ref.rollout.multi_turn.get("save_negative_samples", False)
        max_negative_samples_per_group = config.actor_rollout_ref.rollout.multi_turn.get("max_negative_samples_per_group", 1)
        cls.rollback_manager = RollbackManager(enable_rollback, max_retries, error_patterns, 
                                                save_negative_samples, max_negative_samples_per_group)
        ### DEBUG ###
        print(f"\n{'='*70}")
        print(f"Tool Rollback Configuration:")
        print(f"  - Rollback enabled: {enable_rollback}")
        print(f"  - Max retries per position: {max_retries}")
        print(f"  - Error patterns: {error_patterns}")
        print(f"  - Save negative samples: {save_negative_samples}")
        print(f"  - Max negative samples per group: {max_negative_samples_per_group}")
        print(f"{'='*70}\n")
        ### DEBUG ###

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )
        # Initialize interactions from config file
        cls.interaction_config_file = config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        if cls.interaction_config_file:
            cls.interaction_map: dict[str, BaseInteraction] = cls._initialize_interactions(cls.interaction_config_file)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)
        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data, sampling_params)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data is not None else {}
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields={},
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        
        # Add negative samples to output if any were collected
        if agent_data.negative_samples:
            output.extra_fields["negative_samples"] = agent_data.negative_samples
            
            ### DEBUG ###
            # Log detailed statistics
            error_type_counts = {}
            for neg_sample in agent_data.negative_samples:
                for error_type in neg_sample.get("error_types", []):
                    error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
            
            # Verify all negative samples have score=-1
            all_scores = [ns.get('score', 'N/A') for ns in agent_data.negative_samples]
            scores_correct = all(score == -1 for score in all_scores)
            
            logger.warning(
                f"\n{'='*70}\n"
                f"📊 [NEGATIVE SAMPLES SUMMARY]\n"
                f"  Request ID: {agent_data.request_id}\n"
                f"  Total negative samples from rollback: {len(agent_data.negative_samples)}\n"
                f"  Error types distribution: {error_type_counts}\n"
                f"  Total tool attempts: {agent_data.total_tool_attempts}\n"
                f"  Assistant turns: {agent_data.assistant_turns}\n"
                f"  User turns: {agent_data.user_turns}\n"
                f"  All scores=-1: {scores_correct} (scores: {all_scores})\n"
                f"{'='*70}"
            )
            ### DEBUG ###
        elif self.rollback_manager.save_negative_samples:
            ### DEBUG ###
            logger.info(f"No negative samples collected for request {agent_data.request_id}")
            ### DEBUG ###
        
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    agent_data.messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=agent_data.image_data, return_tensors="pt")
            agent_data.prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            agent_data.prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    agent_data.messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
            )

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Extract tool calls
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

        # Handle interaction if needed
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], tool_position_key: Optional[str] = None
    ) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses with rollback support."""
        # Safety check: disable rollback if too many attempts to prevent infinite loops
        agent_data.total_tool_attempts += 1
        MAX_TOOL_ATTEMPTS_BEFORE_DISABLE = 30
        
        if agent_data.total_tool_attempts > MAX_TOOL_ATTEMPTS_BEFORE_DISABLE and not agent_data.disable_rollback_after_max_retry:
            logger.warning(
                f"⚠️ Total tool attempts ({agent_data.total_tool_attempts}) exceeded {MAX_TOOL_ATTEMPTS_BEFORE_DISABLE}. "
                f"Disabling rollback mechanism to prevent infinite loops."
            )
            agent_data.disable_rollback_after_max_retry = True
        
        # Execute tool calls
        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Determine the key to use for tracking retries
        # If passed from rollback (recursive call), use it to maintain continuity across turns.
        # Otherwise, use current turn index.
        if tool_position_key is None:
            tool_position_key = f"turn_{agent_data.assistant_turns}"

        # Check for rollback-triggering errors
        # We check if rollback is enabled AND (not disabled globally OR we are currently in a retry loop that needs resolution)
        is_retrying = agent_data.retry_counts.get(tool_position_key, 0) > 0
        if self.rollback_manager.enable and (not agent_data.disable_rollback_after_max_retry or is_retrying):
            error_messages, error_types = self._detect_errors(responses, tool_position_key)
            
            # Handle rollback if needed
            if error_messages:
                # Save negative sample BEFORE creating checkpoint (if this is first failure)
                # This ensures we capture the original failed state
                if (agent_data.retry_counts.get(tool_position_key, 0) == 0 and
                    self.rollback_manager.save_negative_samples and 
                    agent_data.negative_samples_count < self.rollback_manager.max_negative_samples_per_group):
                    negative_sample = self._create_negative_sample(agent_data, error_messages, error_types, tool_position_key)
                    agent_data.negative_samples.append(negative_sample)
                    agent_data.negative_samples_count += 1
                    ### DEBUG ###
                    logger.warning(
                        f"💾 [NEGATIVE SAMPLE] Saved {agent_data.negative_samples_count}/{self.rollback_manager.max_negative_samples_per_group} | "
                        f"Position: {tool_position_key} | Errors: {error_types} | "
                        f"Response length: {len(negative_sample['response_ids'])} tokens"
                    )
                    # Print detailed negative sample content to verify error messages are included
                    logger.warning(
                        f"📋 [NEGATIVE SAMPLE DETAIL]\n"
                        f"negative_sample:{negative_sample}"
                    )
                    ### DEBUG ###
                
                # Now create checkpoint (only when actually needed)
                # tool_position_key is already set correctly above
                
                # Record rollback trigger only on first failure (retry_count == 0)
                if agent_data.retry_counts.get(tool_position_key, 0) == 0:
                    await _record_rollback_trigger()
                
                # Check if retry limit exceeded OR rollback is disabled globally (force failure for existing retry)
                if not self.rollback_manager.can_retry(agent_data.retry_counts, tool_position_key) or agent_data.disable_rollback_after_max_retry:
                    # Record rollback final failure when max retry is reached
                    # Only record if not already recorded as recovered
                    if tool_position_key not in agent_data.rollback_recovered_turns:
                        await _record_rollback_final_failure()
                    # logger.warning(
                    #     f"⚠️ Tool retry limit reached at {tool_position_key}. "
                    #     f"Notifying model about tool execution failure."
                    # )
                    # Give model a final notification instead of direct termination
                    return await self._handle_max_retry_exceeded(
                        agent_data, responses, error_messages, error_types, tool_call_names, sampling_params
                    )
                
                checkpoint = self.rollback_manager.create_checkpoint(agent_data)
                rollback_result = await self._handle_rollback(
                    agent_data, checkpoint, tool_position_key, error_messages, error_types, sampling_params
                )
                if rollback_result is not None:
                    return rollback_result
            else:
                # Log successful tool execution (no errors detected)
                # Use the same key logic
                retry_count = agent_data.retry_counts.get(tool_position_key, 0)
                
                # Record rollback recovery if this turn had previous failures and hasn't been recorded yet
                if retry_count > 0 and tool_position_key not in agent_data.rollback_recovered_turns:
                    agent_data.rollback_recovered_turns.add(tool_position_key)
                    await _record_rollback_recovery()
                    ### DEBUG ###
                    logger.warning(
                        f"✅ [ROLLBACK SUCCESS] Position: {tool_position_key} | "
                        f"Recovered after {retry_count} retry(ies) | "
                        f"Total attempts: {agent_data.total_tool_attempts}"
                    )
                    ### DEBUG ###

        # No rollback needed - process tool responses normally
        return await self._process_tool_responses(agent_data, responses, tool_call_names)
    
    def _detect_errors(self, responses: list[tuple], tool_position_key: str) -> tuple[list[str], list[str]]:
        """Detect rollback-triggering errors in tool responses.
        
        Returns:
            tuple[list[str], list[str]]: (error_messages, error_types)
        """
        error_messages = []
        error_types = []
        
        for i, (tool_response, tool_reward, _) in enumerate(responses):
            error_text = tool_response.text or ""
            should_rollback, error_type = self.rollback_manager.should_rollback(error_text)
            if should_rollback:
                error_messages.append(error_text)
                error_types.append(error_type)
                # logger.warning(
                #     f"[Retry {current_retry}/{self.rollback_manager.max_retries}] "
                #     f"Tool call #{i} failed with error type: {error_type}"
                # )
        return error_messages, error_types
    
    async def _handle_rollback(
        self, 
        agent_data: AgentData, 
        checkpoint: dict[str, Any],
        tool_position_key: str,
        error_messages: list[str],
        error_types: list[str],
        sampling_params: dict[str, Any]
    ) -> Optional[AgentState]:
        """Handle the rollback process. Returns AgentState if rollback is triggered, None otherwise."""
        if not error_messages:
            return None
            
        retry_count = self.rollback_manager.increment_retry(agent_data.retry_counts, tool_position_key)
        ### DEBUG ###
        logger.warning(
            f"🔄 Rollback initiated at {tool_position_key} | "
            f"Error types: {error_types} | "
            f"Retry: {retry_count}/{self.rollback_manager.max_retries}"
        )
        ### DEBUG ###

        # Step 1: Append error feedback to context
        error_feedback = self.rollback_manager.format_error_feedback(error_messages)
        error_message = {"role": "user", "content": error_feedback}
        agent_data.messages.append(error_message)

        # Step 2: Encode error feedback
        error_prompt_ids = await self._encode_error_feedback(agent_data, error_message)
        agent_data.prompt_ids += error_prompt_ids
        agent_data.response_mask += [0] * len(error_prompt_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(error_prompt_ids)

        # Step 3: Let LLM regenerate tool calls
        new_state = await self._handle_generating_state(agent_data, sampling_params, ignore_termination=True)

        if new_state == AgentState.TERMINATED or not agent_data.tool_calls:
            # Unable to generate valid tool calls - restore checkpoint and continue processing
            # to let the normal flow handle this (will trigger retry or max retry logic)
            # logger.error(
            #     f"❌ Failed to regenerate tool calls at {tool_position_key} after {retry_count} retries. "
            #     f"Will attempt next retry or handle max retry."
            # )
            
            # Record failure as we are aborting the rollback attempt
            if tool_position_key not in agent_data.rollback_recovered_turns:
                await _record_rollback_final_failure()

            self.rollback_manager.restore_checkpoint(agent_data, checkpoint)
            # Return TERMINATED to stop this retry attempt - the recursive call will handle next attempt
            return AgentState.TERMINATED

        ### DEBUG ###
        logger.info(f"✅ Successfully regenerated {len(agent_data.tool_calls)} tool call(s) at {tool_position_key}")
        ### DEBUG ###

        # Step 4: Restore checkpoint
        self.rollback_manager.restore_checkpoint(agent_data, checkpoint)

        # Step 5: Recursive retry
        return await self._handle_processing_tools_state(agent_data, sampling_params, tool_position_key=tool_position_key)
    
    async def _handle_max_retry_exceeded(
        self,
        agent_data: AgentData,
        responses: list[tuple],
        error_messages: list[str],
        error_types: list[str],
        tool_call_names: list[str],
        sampling_params: dict[str, Any]
    ) -> AgentState:
        """Handle max retry exceeded: notify model instead of direct termination.
        
        For math problems, the model can:
        1. Acknowledge the computational limitation
        2. Break down the problem into simpler steps
        3. Use alternative approaches or approximations
        """
        # Extract the last tool call's error information (iterate in reverse to get the last one)
        last_error_detail = None
        
        for i in range(len(responses) - 1, -1, -1):  # Iterate backwards
            tool_response, tool_reward, _ = responses[i]
            error_text = tool_response.text or ""
            should_rollback, error_type = self.rollback_manager.should_rollback(error_text)
            if should_rollback:
                tool_name = tool_call_names[i] if i < len(tool_call_names) else "unknown"
                # Found the last failed tool call, extract its complete error message
                last_error_detail = f"Tool '{tool_name}' error: {error_text}"
                break  # Stop after finding the last error
        
        # Create failure notification with the last error detail
        if last_error_detail:
            failure_message = (
                f"Tool call failure happened after {self.rollback_manager.max_retries} attempts. The last error was:\n\n"
                f"{last_error_detail}\n\n"
                f"The current approach may have technical limitations. "
                f"Consider: (1) breaking into smaller steps, or"
                f"(2) using alternative methods."
            )
        else:
            # Fallback if no error detail found
            failure_message = (
                f"Tool call failure happened after {self.rollback_manager.max_retries} attempts. "
                f"The current approach may have technical limitations."
            )
        
        # Disable rollback for subsequent tool calls to prevent infinite loops
        agent_data.disable_rollback_after_max_retry = True
        # logger.warning(
        #     f"⚠️ Rollback disabled for remaining tool calls to prevent infinite retry loops. "
        #     f"failure_message: {failure_message}"
        # )
        
        # Add as tool response to maintain conversation flow
        final_notification = {"role": "tool", "content": failure_message}
        agent_data.messages.append(final_notification)
        
        # Encode notification
        if self.processor is not None:
            raw_notification = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    [final_notification],
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_notification], images=None, return_tensors="pt")
            notification_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            notification_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    [final_notification], add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                ),
            )
            notification_ids = notification_ids[len(self.system_prompt) :]
        
        # Update agent state
        agent_data.prompt_ids += notification_ids
        agent_data.response_mask += [0] * len(notification_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(notification_ids)
        agent_data.user_turns += 1
        
        # Check length limit
        if len(agent_data.response_mask) >= self.response_length:
            logger.warning("Response length limit reached after max retry notification")
            return AgentState.TERMINATED
        
        # Let model continue and decide how to handle the failure
        # logger.info(f"Allowing model to continue after tool failure at turn {agent_data.assistant_turns}")
        return AgentState.GENERATING
    
    async def _encode_error_feedback(self, agent_data: AgentData, error_message: dict[str, Any]) -> list[int]:
        """Encode error feedback message to token ids."""
        if self.processor is not None:
            raw_error_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    [error_message],
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_error_prompt], images=None, return_tensors="pt")
            return model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            error_prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    [error_message], add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                ),
            )
            return error_prompt_ids[len(self.system_prompt) :]
    
    async def _process_tool_responses(
        self, 
        agent_data: AgentData, 
        responses: list[tuple],
        tool_call_names: list[str]
    ) -> AgentState:
        """Process tool responses and update agent state."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []

        # Process tool responses and update multi_modal_data
        for tool_response, tool_reward, _ in responses:
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        agent_data.messages.extend(add_messages)
        # Update prompt with tool responses
        if self.processor is not None:
            raw_tool_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            # Use only the new images from this turn for processing tool responses
            current_images = new_images_this_turn if new_images_this_turn else None  # Using local variable
            model_inputs = self.processor(text=[raw_tool_response], images=current_images, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            if self.tool_parser_name == "gpt-oss":
                logger.info("manually format tool responses for gpt-oss")
                # Format tool responses manually
                tool_response_texts = []
                for i, tool_msg in enumerate(add_messages):
                    actual_tool_name = tool_call_names[i]
                    formatted = format_gpt_oss_tool_response_manually(tool_msg["content"], actual_tool_name)
                    tool_response_texts.append(formatted)

                tool_response_text = add_generation_prompt_for_gpt_oss("".join(tool_response_texts))
                response_ids = await self.loop.run_in_executor(
                    None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
                )
            else: 
                response_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
                )
                response_ids = response_ids[len(self.system_prompt) :]
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]
        agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        if self.processor is not None:
            raw_user_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_user_response], images=None, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
            )
        response_ids = response_ids[len(self.system_prompt) :]

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # double check prompt
        # Check termination condition
        if should_terminate_sequence:
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    def _create_negative_sample(self, agent_data: AgentData, error_messages: list[str], 
                                error_types: list[str], tool_position_key: str) -> dict[str, Any]:
        """Create a negative sample from current failed trajectory.
        
        Returns a dictionary containing the failed trajectory information that can be used
        for training as a negative sample.
        """
        return {
            "prompt_ids": list(agent_data.prompt_ids),
            "response_ids": agent_data.response_ids.copy() if agent_data.response_ids else [],
            "response_mask": list(agent_data.response_mask),
            "response_logprobs": list(agent_data.response_logprobs) if agent_data.response_logprobs else None,
            "error_messages": error_messages,
            "error_types": error_types,
            "tool_position": tool_position_key,
            "assistant_turns": agent_data.assistant_turns,
            "user_turns": agent_data.user_turns,
            "tool_calls": [{
                "name": tc.name,
                "arguments": tc.arguments,
                "id": getattr(tc, "id", None)
            } for tc in agent_data.tool_calls],
            "score": -1,  # Negative samples always get score=-1
        }
    
    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any]
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        tool_execution_response = None
        tool_reward = None
        res = None
        stat_recorded = False  # Flag to prevent duplicate recording
        
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json_repair.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, tool_reward, res = await tool.execute(instance_id, tool_args)
        except Exception as e:
            if "'str' object has no attribute 'get'" in str(e):
                logger.warning(f"tool call format is wrong")
                await _record_tool_stat(False)
                stat_recorded = True
                return (
                    ToolResponse(
                        text=f"Tool call failure: tool call format is wrong, please make sure to generate correct json-format tool call arguments.",
                    ),
                    0.0,
                    {},
                )
            logger.warning(f"Error when executing tool: {e}")
            await _record_tool_stat(False)
            stat_recorded = True
            return (
                ToolResponse(
                    text=f"Tool call failure. Error when executing tool: {e}",
                ),
                0.0,
                {},
            )
        finally:
            # Record tool call statistics based on response content (only if not already recorded)
            if not stat_recorded and tool_execution_response is not None:
                response_text = tool_execution_response.text or ""
                is_success = "Tool call success" in response_text
                await _record_tool_stat(is_success)
            
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}
        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    @classmethod
    def _initialize_interactions(cls, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        logger.info(f"Initialize interactions from configuration: interaction_map: {list(interaction_map.keys())}")
        return interaction_map

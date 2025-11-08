# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

TOOL_NAME_ALIASES: dict[str, str] = {
    "code_interation": "code_interpreter",
    "code_interpriter": "code_interpreter",
}


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
        parser_kwargs = {}
        enable_repair = getattr(
            config.actor_rollout_ref.rollout.multi_turn,
            "tool_parser_enable_repair",
            None,
        )
        if enable_repair is not None:
            parser_kwargs["enable_repair"] = enable_repair
        cls.tool_parser = ToolParser.get_tool_parser(
            config.actor_rollout_ref.rollout.multi_turn.format,
            cls.tokenizer,
            **parser_kwargs,
        )
        print(f"Initialized tools: {cls.tools}")

        feedback_override = getattr(
            config.actor_rollout_ref.rollout.multi_turn,
            "tool_parser_enable_feedback",
            None,
        )
        if feedback_override is None:
            env_feedback = os.getenv("VERL_TOOL_PARSER_ENABLE_FEEDBACK")
            cls.tool_parser_enable_feedback = (
                True
                if env_feedback is None
                else env_feedback.lower() not in {"0", "false", "no"}
            )
        else:
            if isinstance(feedback_override, str):
                cls.tool_parser_enable_feedback = feedback_override.lower() not in {"0", "false", "no"}
            else:
                cls.tool_parser_enable_feedback = bool(feedback_override)

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )
# sd
    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        # if logger.isEnabledFor(logging.INFO):
        #     logger.info("raw_prompt messages:\n%s", json.dumps(messages, ensure_ascii=False, indent=2))
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        metrics = {}
        request_id = uuid4().hex
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        response_mask, response_logprobs = [], []
        tools_kwargs = kwargs.get("tools_kwargs", {})

        user_turns, assistant_turns = 0, 0
        parser_error_messages: list[str] = []
        parser_error_count = 0
        parser_attempts = 0
        parsed_tool_call_count = 0
        while True:
            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=image_data
                )
            response_ids = output.token_ids
            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)
            if output.log_probs:
                response_logprobs += output.log_probs
            assistant_turns += 1

            # reach max response length
            if len(response_mask) >= self.response_length:
                break

            # reach max assistant turns
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                break

            # reach max user turns
            if self.max_user_turns and user_turns >= self.max_user_turns:
                break

            # Parse tool calls from LLM response
            _, tool_calls, parser_errors = await self.tool_parser.extract_tool_calls(response_ids)
            if tool_calls or parser_errors:
                parser_attempts += 1
            if parser_errors:
                parser_error_messages.extend(parser_errors)
                parser_error_count += len(parser_errors)
            if parser_attempts and parser_attempts % 200 == 0 and logger.isEnabledFor(logging.INFO):
                error_rate = parser_error_count / parser_attempts if parser_attempts else 0.0
                sample_error = parser_error_messages[-1][:120] if parser_error_messages else ""
                logger.info(
                    "Tool parser stats (recent %d attempts): total_errors=%d detected_calls=%d error_rate=%.3f last_error='%s'",
                    parser_attempts,
                    parser_error_count,
                    parsed_tool_call_count,
                    error_rate,
                    sample_error,
                )
            if tool_calls:
                parsed_tool_call_count += len(tool_calls)

            tool_messages: list[dict[str, Any]] = []
            if parser_errors and self.tool_parser_enable_feedback:
                tool_messages.extend(
                    {
                        "role": "tool",
                        "content": (
                            "[tool_parser_error] "
                            f"{error}. Please return a JSON object with keys 'name' and 'arguments' "
                            "inside <tool_call>...</tool_call>."
                        ),
                    }
                    for error in parser_errors
                )

            # No valid tool calls and no parser feedback -> exit loop
            if not tool_calls and not tool_messages:
                break

            # call tools
            tool_responses = []
            if tool_calls:
                tasks = [self._call_tool(tool_call, tools_kwargs) for tool_call in tool_calls[: self.max_parallel_calls]]
                with simple_timer("tool_calls", metrics):
                    tool_responses = await asyncio.gather(*tasks)
                if any(isinstance(item, Exception) for item in tool_responses):
                    break

            # Extract messages and update multi_modal_data
            new_images_this_turn: list[Any] = []
            for tool_response in tool_responses:
                if tool_response.image or tool_response.video:
                    content = []
                    if tool_response.image:
                        content.append({"type": "image"})
                    if tool_response.video:
                        content.append({"type": "video"})
                    if tool_response.text:
                        content.append({"type": "text", "text": tool_response.text})
                    message = {"role": "tool", "content": content}
                else:
                    message = {"role": "tool", "content": tool_response.text or ""}

                tool_messages.append(message)

                if tool_response.image:
                    if image_data is None:
                        image_data = []
                    elif not isinstance(image_data, list):
                        image_data = [image_data]

                    if isinstance(tool_response.image, list):
                        image_data.extend(tool_response.image)
                        new_images_this_turn.extend(tool_response.image)
                    else:
                        image_data.append(tool_response.image)
                        new_images_this_turn.append(tool_response.image)

                if tool_response.video:
                    logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                    raise NotImplementedError(
                        "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                    )

            if not tool_messages:
                continue

            tool_response_ids = await self._encode_tool_messages(tool_messages, new_images_this_turn)

            # NOTE: last turn should not be user turn, or the EOS token reward
            # can't be propagated to previous token in GAE.
            if len(response_mask) + len(tool_response_ids) >= self.response_length:
                break

            prompt_ids += tool_response_ids
            response_mask += [0] * len(tool_response_ids)
            if response_logprobs:
                response_logprobs += [0.0] * len(tool_response_ids)
            user_turns += 1

        response_ids = prompt_ids[-len(response_mask) :]
        prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]

        multi_modal_data = {"image": image_data} if image_data is not None else {}

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            num_turns=user_turns + assistant_turns + 1,
            metrics=metrics,
        )
        output.extra_fields.update(
            {
                "tool_parser_error_count": parser_error_count,
                "tool_parser_attempts": parser_attempts,
                "tool_parser_error_messages": parser_error_messages,
                "tool_parser_detected_tool_calls": parsed_tool_call_count,
            }
        )
        return output

    async def _encode_tool_messages(
        self,
        tool_messages: list[dict[str, Any]],
        new_images_this_turn: list[Any],
    ) -> list[int]:
        if not tool_messages:
            return []

        if self.processor is not None:
            raw_tool_response = await self.loop.run_in_executor(
                None,
                lambda messages=tool_messages: self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                ),
            )
            current_images = new_images_this_turn if new_images_this_turn else None
            model_inputs = self.processor(text=[raw_tool_response], images=current_images, return_tensors="pt")
            tool_response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            tool_response_ids = await self.loop.run_in_executor(
                None,
                lambda messages=tool_messages: self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                ),
            )

        return tool_response_ids[len(self.system_prompt) :]

    async def _call_tool(self, tool_call: FunctionCall, tools_kwargs: dict[str, Any]) -> ToolResponse:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = TOOL_NAME_ALIASES.get(tool_call.name, tool_call.name)
            if tool_name != tool_call.name:
                logger.debug("Normalizing tool name %s -> %s", tool_call.name, tool_name)
            tool_args = json.loads(tool_call.arguments)
            if not isinstance(tool_args, dict):
                raise ValueError(
                    f"Tool arguments must be a JSON object, got {type(tool_args).__name__}: {tool_call.arguments[:200]}"
                )
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, _, _ = await tool.execute(instance_id, tool_args)
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return ToolResponse(
                text=f"Error when executing tool: {e}",
            )
        finally:
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

        return ToolResponse(**tool_response_kwargs)

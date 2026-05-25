"""
Unified benchmark script for running the CV Agent on multiple datasets (e.g., MME, CV-Bench)
using a pluggable dataset loader architecture.
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import ray
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import get_client, propagate_attributes

# --- Imports from your project structure ---
from cv_agent.benchmark_loaders import get_dataset_loader
from cv_agent.core.builder import GraphBuilder
from cv_agent.core.registries import chat_model_registry
from cv_agent.utils.config import get_invocation_config
from cv_agent.utils.grader import Grade, grade_answer
from cv_agent.utils.storage import pil_to_base64_uri, upload_pil_image_to_minio
from cv_agent.utils.trajectory_logger import TrajectoryLogger

# --- Global Logger ---
logger = logging.getLogger(__name__)


def setup_logging(worker_id: int, log_dir: str) -> None:
    """Configures logging for a ray worker to write to its own file."""
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"worker_{worker_id}.log")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(f"[%(asctime)s] [Worker-{worker_id}] [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    logger.info("Logging configured. Saving logs to %s", log_file_path)


def extract_answer(message_content: str) -> str:
    """Extracts content from <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", message_content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    thought_action_pattern = r"Thought:.*"
    cleaned_content = re.sub(thought_action_pattern, "", message_content, flags=re.DOTALL).strip()
    if not cleaned_content:
        return message_content
    return cleaned_content


async def run_agent_sample_with_retry(
    question: str,
    initial_image: Image.Image,
    agent_executor,
    sample_id: str,
    invoke_config: dict,
    state_overrides: dict,
    turn_info: str = "Sample",
) -> tuple[str, str, dict, dict]:
    """Returns (final_answer, raw_content, tool_usage, trajectory_data).

    *trajectory_data* keys: trajectory_steps, total_input_tokens,
    total_output_tokens, image_url, img_width, img_height.
    """
    """
    Wrapper to run agent sample with retry logic.
    We separate this from the core logic to allow the retry decorator to work cleanly.
    """

    # Retries up to 3 times if AgentExecutionError is raised.

    class AgentExecutionError(Exception):
        """Exception raised when the agent fails to produce a valid answer (e.g., recursion limit)."""

        pass

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(AgentExecutionError),
        reraise=False,  # If all retries fail, return the result of the last attempt (which will be the exception raised)
    )
    async def run_agent_sample_core(
        question: str,
        initial_image: Image.Image,
        agent_executor,
        sample_id: str,
        invoke_config: dict,
        state_overrides: dict,
        turn_info: str = "Sample",
    ) -> tuple[str, str, dict, dict]:
        """
        Core logic for running the CV agent. This function RAISES exceptions on failure
        instead of returning error strings, allowing the wrapper to catch and retry.
        """
        public_minio_url = ""
        img_width, img_height = 0, 0

        try:
            public_minio_url, img_width, img_height = await upload_pil_image_to_minio(
                initial_image, sample_id
            )
            if not public_minio_url:
                raise ValueError("Failed to upload initial image to Minio.")
        except Exception as e:
            logger.error("[%s] Error during image upload: %s", turn_info, e)
            raise ValueError(f"Failed to upload image: {e}") from e

        initial_state = {
            "question": question,
            "original_figure_url": public_minio_url,
            "current_turn": 1,
            "max_turns": 10,
            "messages": [],
            "image_dimensions": {public_minio_url: (img_width, img_height)},
            "prefix": sample_id,
            "url_map": {public_minio_url: public_minio_url},
            "direct_reasoning_result": "",
            "tool_usage": {},
            # trajectory tracking
            "trajectory_steps": [],
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "pending_thinking": "",
            # Belief annotation
            "pending_belief": {},
            "episode_context": {},
            "enable_belief_annotation": False,
            # SkillGuard intervention log (populated when enable_skillguard=True)
            "skillguard_log": [],
            # VoI Gate log (populated when enable_voi_gate=True)
            "gate_log": [],
        }
        if state_overrides:
            initial_state.update(state_overrides)

        final_answer = "Error: No final answer found."
        final_state_messages = []
        final_tool_usage = {}

        try:
            # async for event in agent_executor.astream_events(
            #     initial_state, version="v1", config=invoke_config
            # ):
            #     kind = event["event"]
            #
            #     if kind == "on_chat_model_stream":
            #         chunk = event["data"]["chunk"]
            #         if chunk.content:
            #             logger.debug(chunk.content, end="", flush=True)
            #     elif kind == "on_tool_start":
            #         logger.info("[%s] --- TOOL START: %s ---", turn_info, event["name"])
            #         logger.debug("Inputs: %s", event["data"].get("input"))
            #     elif kind == "on_tool_end":
            #         tool_output = event["data"].get("output")
            #         logger.info("[%s] --- TOOL END: %s ---", turn_info, event["name"])
            #         logger.debug("Output: %s...", str(tool_output)[:200])
            #     elif kind == "on_chain_end":
            #         final_response = event["data"].get("output")
            #         if final_response and "messages" in final_response:
            #             final_state_messages = final_response["messages"]
            #
            #         if "tool_usage" in final_response:
            #             final_tool_usage = final_response["tool_usage"]
            #
            #     elif kind == "on_error":
            #         # Catch stream errors (like recursion limit)
            #         error_msg = event["data"]["error"]
            #         logger.warning("[%s] Error in stream: %s", turn_info, error_msg)
            #         # RAISE here to trigger retry!
            #         raise AgentExecutionError(f"Agent stream error: {error_msg}")
            final_state = await agent_executor.ainvoke(initial_state,config=invoke_config)

            final_state_messages = final_state["messages"]
            if final_state_messages:
                final_content = final_state_messages[-1].content
                final_answer = extract_answer(final_content)

                # Check for hallucinated tool calls in the answer itself or missing answer
                if "Error: Agent finished without a final answer tag" in final_answer:
                    raise AgentExecutionError("Agent finished without <answer> tag.")

            else:
                # If we finished the stream but found no messages, that's a failure.
                raise AgentExecutionError("Stream completed but no state messages found.")

        except AgentExecutionError:
            raise  # Pass through to decorator
        except Exception as e:
            # Catch other unexpected errors during execution
            logger.error("[%s] Agent run failed with exception: %s", turn_info, e)
            raise AgentExecutionError(f"Agent run failed: {e}")

        final_tool_usage = final_state.get("tool_usage", {})

        trajectory_data = {
            "trajectory_steps": final_state.get("trajectory_steps", []),
            "total_input_tokens": final_state.get("total_input_tokens", 0),
            "total_output_tokens": final_state.get("total_output_tokens", 0),
            "image_url": public_minio_url,
            "img_width": img_width,
            "img_height": img_height,
            "episode_context": final_state.get("episode_context", {}),
            "gate_log": final_state.get("gate_log", []),
        }
        return final_answer, final_content, final_tool_usage, trajectory_data

    try:
        return await run_agent_sample_core(
            question,
            initial_image,
            agent_executor,
            sample_id,
            invoke_config,
            state_overrides,
            turn_info,
        )
    except AgentExecutionError as e:
        logger.warning(f"[{turn_info}] Agent execution failed (attempt will retry): {e}")
        raise e  # Re-raise to trigger tenacity retry
    except Exception as e:
        # Non-retriable errors (like Minio upload failure) just fail immediately
        logger.error(f"[{turn_info}] Critical error (no retry): {e}")
        return f"Error: {e}", "", {}, {}


DIRECT_SYSTEM_PROMPT = """You are a specialized Computer Vision model.
Your task is to answer the user's question about the image.

First, you must provide a step-by-step chain of thought on how you arrived at the answer.
After your reasoning, 
you MUST provide the final answer, and *only* the final answer, wrapped in <answer> tags.

### Example
This is a blue glove. 
The shiny texture suggests it is made of rubber or latex, not cotton or leather.
<answer>A</answer>

### Final Answer Format
You MUST provide your final answer, and *only* the final answer, wrapped in <answer> tags.

**Example for a general question:**
<answer>There are 3 cats in the image.</answer>

**Example for a Multiple-Choice (A, B, C, D) question:**
<answer>A</answer>
"""
#
#
# async def run_direct_model_sample(
#     llm: BaseChatModel,
#     question: str,
#     initial_image: Image.Image,
#     sample_id: str,
#     turn_info: str = "Sample",
# ) -> (str, str):
#     """
#     Runs the base LLM (no tools) for a single question and image.
#     """
#
#     # Define which exceptions should trigger a retry.
#     # For remote LLM/Tool calls, connection errors or rate limits are common.
#     class RemoteError(Exception):
#         """Custom exception for errors during remote LLM inference."""
#
#         pass
#
#     # Retry decorator definition:
#     # Stop after 10 attempts, waiting 2s, 4s, etc., between retries.
#     @retry(
#         stop=stop_after_attempt(10),
#         wait=wait_exponential(multiplier=1, min=2, max=60),
#         retry=retry_if_exception_type(RemoteError),
#         reraise=True,  # Ensure the original exception type is raised if all attempts fail
#     )
#     async def resilient_llm_call(llm: BaseChatModel, messages_to_send: list):
#         """Wrapper for LLM calls with retry logic."""
#         try:
#             response = await llm.ainvoke(messages_to_send)
#             return response
#         except Exception as e:
#             # Check for common network/API errors that should be retried
#             # In a real setup, you'd check for specific status codes (e.g., 500, 429)
#             if (
#                 "timeout" in str(e).lower()
#                 or "connection error" in str(e).lower()
#                 or "rate limit" in str(e).lower()
#             ):
#                 logger.warning(
#                     f"Temporary LLM error detected ({e.__class__.__name__}). Retrying..."
#                 )
#                 raise RemoteError(f"Retriable LLM Error: {e}")
#             else:
#                 # For unrecoverable errors (like bad API key, validation error), do not retry
#                 raise
#
#     try:
#         base64_data_uri = pil_to_base64_uri(initial_image)
#     except Exception as e:
#         logger.error("[%s] Error during Base64 encoding: %s", turn_info, e)
#         # 解决 Critical failure: too many values to unpack (expected 2)
#         return f"Error: Base64 encoding failed: {e}", ""
#
#     try:
#         system_message = SystemMessage(content=DIRECT_SYSTEM_PROMPT)
#
#         human_content = [
#             {"type": "text", "text": question},
#             {"type": "image_url", "image_url": {"url": base64_data_uri}},  # Use base64
#         ]
#         human_message = HumanMessage(content=human_content)
#
#         messages_to_send = [system_message, human_message]
#
#         logger.info("[%s] Calling direct model...", turn_info)
#         response = await resilient_llm_call(llm, messages_to_send)
#
#         return extract_answer(response.content), response.content
#
#     except Exception as e:
#         logger.error("[%s] Direct model run failed: %s", turn_info, e)
#         return f"Error: Direct model run failed: {e}"


async def run_direct_model_sample(
    llm: BaseChatModel,
    question: str,
    initial_image: Image.Image,
    sample_id: str,
    turn_info: str = "Sample",
) -> (str, str):
    """
    Runs the base LLM (no tools) for a single question and image.
    """

    # Define which exceptions should trigger a retry.
    # For remote LLM/Tool calls, connection errors or rate limits are common.
    class RemoteError(Exception):
        """Custom exception for errors during remote LLM inference."""

        pass

    # Retry decorator definition:
    # Stop after 10 attempts, waiting 2s, 4s, etc., between retries.
    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(RemoteError),
        reraise=True,  # Ensure the original exception type is raised if all attempts fail
    )
    async def resilient_llm_call(llm: BaseChatModel, messages_to_send: list):
        """Wrapper for LLM calls with retry logic."""
        try:
            response = await llm.ainvoke(messages_to_send)

            # --- [新增逻辑] 检查响应是否为空 ---
            content = response.content
            # 如果 content 为 None，或者 content 是字符串且去空格后为空
            if not content or (isinstance(content, str) and not content.strip()):
                logger.warning(
                    f"[{turn_info}] Empty response received from LLM. Triggering retry..."
                )
                # 主动抛出 RemoteError，让装饰器捕获并重试
                raise RemoteError("Empty response from LLM")
            # --------------------------------

            return response

        # --- [新增逻辑] 优先捕获 RemoteError 并透传 ---
        # 必须显式捕获 RemoteError 并重新抛出，防止被下面的通用 Exception 吞掉
        except RemoteError:
            raise
        # -------------------------------------------

        except Exception as e:
            # Check for common network/API errors that should be retried
            # In a real setup, you'd check for specific status codes (e.g., 500, 429)
            if (
                "timeout" in str(e).lower()
                or "connection error" in str(e).lower()
                or "rate limit" in str(e).lower()
                or "service unavailable" in str(e).lower()
                or "500" in str(e)
                or "503" in str(e)
            ):
                logger.warning(
                    f"Temporary LLM error detected ({e.__class__.__name__}): {e}. Retrying..."
                )
                raise RemoteError(f"Retriable LLM Error: {e}")
            else:
                # For unrecoverable errors (like bad API key, validation error), do not retry
                raise

    try:
        base64_data_uri = pil_to_base64_uri(initial_image)
    except Exception as e:
        logger.error("[%s] Error during Base64 encoding: %s", turn_info, e)
        # 解决 Critical failure: too many values to unpack (expected 2)
        return f"Error: Base64 encoding failed: {e}", ""

    try:
        system_message = SystemMessage(content=DIRECT_SYSTEM_PROMPT)

        human_content = [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": base64_data_uri}},  # Use base64
        ]
        human_message = HumanMessage(content=human_content)

        messages_to_send = [system_message, human_message]

        logger.info("[%s] Calling direct model...", turn_info)
        # 这里调用内部定义的带重试逻辑的函数
        response = await resilient_llm_call(llm, messages_to_send)

        return extract_answer(response.content), response.content

    except Exception as e:
        logger.error("[%s] Direct model run failed: %s", turn_info, e)
        return f"Error: Direct model run failed: {e}", ""


async def run_benchmark_task(worker_id: int, data_indices: list, args: argparse.Namespace):
    """
    The main async task for a Ray worker.
    Loads models/data and processes a chunk of the dataset.
    """
    setup_logging(worker_id, args.log_dir)
    logger.info("Ray worker %d starting for %d samples.", worker_id, len(data_indices))

    cfg_path = args.config_path
    if cfg_path is None:
        # Get base dir (script is in src/cv_agent, go up 2 levels)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(script_dir, "..", "..")  # To project root

        if args.mode == "agent":
            if args.model == "qwen-30b":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b.yaml")
            elif args.model == "qwen-30b-skillguard":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_skillguard.yaml")
            elif args.model == "qwen-30b-belief":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_belief.yaml")
            elif args.model == "qwen-30b-voi-gate":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_voi_gate.yaml")
            elif args.model == "qwen-30b-info-gain-logprobs":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_info_gain_logprobs.yaml")
            elif args.model == "qwen-30b-logprobs":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_logprobs.yaml")
            elif args.model == "qwen-30b-logprobs-ep2":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_logprobs_ep2.yaml")
            elif args.model == "qwen-30b-ep2":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_ep2.yaml")
            elif args.model == "qwen-30b-probe-cvbench":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_cvbench.yaml")
            elif args.model == "qwen-30b-probe-hrbench4k":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_hrbench4k.yaml")
            elif args.model == "qwen-30b-probe-hrbench8k":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_hrbench8k.yaml")
            elif args.model == "qwen-30b-probe-mme":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_mme.yaml")
            elif args.model == "qwen-30b-probe-crossdomain-q3":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_crossdomain_q3.yaml")
            elif args.model == "qwen-30b-probe-crossdomain-q3-ep1":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_crossdomain_q3_ep1.yaml")
            elif args.model == "qwen-30b-probe-cross-pv1":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_crossdomain_pv1.yaml")
            elif args.model == "qwen-30b-probe-cross-pv2":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_crossdomain_pv2.yaml")
            elif args.model == "qwen-30b-probe-cross-pv3":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_crossdomain_pv3.yaml")
            elif args.model == "qwen-30b-probe-v4":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_v4.yaml")
            elif args.model == "qwen-30b-probe-v4-ep1":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_v4_ep1.yaml")
            elif args.model == "qwen-30b-probe-v2-qwen-frozen":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_v2_qwen_frozen.yaml")
            elif args.model == "qwen-30b-probe-v2-qwen-frozen-ep1":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_v2_qwen_frozen_ep1.yaml")
            elif args.model == "qwen-30b-probe-qwen-frozen":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_qwen_frozen.yaml")
            elif args.model == "qwen-30b-probe-qwen-frozen-ep1":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_qwen_frozen_ep1.yaml")
            elif args.model == "qwen-30b-probe-qwenft-lora":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_qwenft_lora.yaml")
            elif args.model == "qwen-30b-probe-cross-pv3-ep1":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_crossdomain_pv3_ep1.yaml")
            elif args.model == "qwen-30b-probe-cross-pv3-t60":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_crossdomain_pv3_t60.yaml")
            elif args.model == "qwen-30b-probe-cross-pv3-t60-ep1":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_crossdomain_pv3_t60_ep1.yaml")
            elif args.model == "qwen-30b-probe-cross-pv3-t70":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_crossdomain_pv3_t70.yaml")
            elif args.model == "qwen-30b-probe-cross-pv3-t70-ep1":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_crossdomain_pv3_t70_ep1.yaml")
            elif args.model == "qwen-30b-probe-cross-t30":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_crossdomain_t30.yaml")
            elif args.model == "qwen-30b-probe-cross-t40":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_crossdomain_t40.yaml")
            elif args.model == "qwen-30b-probe-cross-t50":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_crossdomain_t50.yaml")
            elif args.model == "qwen-30b-probe-cross-t60":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_crossdomain_t60.yaml")
            elif args.model == "qwen-30b-probe-cross-t70":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_crossdomain_t70.yaml")
            elif args.model == "qwen-30b-probe-crossq3-t30":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_crossdomain_qwen3_t30.yaml")
            elif args.model == "qwen-30b-probe-crossq3-t40":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_crossdomain_qwen3_t40.yaml")
            elif args.model == "qwen-30b-probe-crossq3-t50":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_crossdomain_qwen3_t50.yaml")
            elif args.model == "qwen-30b-probe-crossq3-t60":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_crossdomain_qwen3_t60.yaml")
            elif args.model == "qwen-30b-probe-crossq3-t70":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_crossdomain_qwen3_t70.yaml")
            elif args.model == "qwen-30b-probe-v3-skipsafe-ep1":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_v3_skipsafe_ep1.yaml")
            elif args.model == "qwen-30b-probe-v3-skipsafe-qwen-c001":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_v3_skipsafe_qwen_c001.yaml")
            elif args.model == "qwen-30b-probe-v3-skipsafe-qwen-c001-ep1":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_v3_skipsafe_qwen_c001_ep1.yaml")
            elif args.model == "qwen-30b-probe-v3-skipsafe-qwen":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_v3_skipsafe_qwen.yaml")
            elif args.model == "qwen-30b-probe-v3-skipsafe-qwen-ep1":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_v3_skipsafe_qwen_ep1.yaml")
            elif args.model == "qwen-235b-probe-skipsafe-qwen":
                cfg_path = os.path.join(base_dir, "configs", "qwen235b_probe_skipsafe_qwen.yaml")
            elif args.model == "qwen-235b-probe-skipsafe-minilm":
                cfg_path = os.path.join(base_dir, "configs", "qwen235b_probe_skipsafe_minilm.yaml")
            elif args.model == "qwen-30b-probe-v3-skipsafe":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_v3_skipsafe.yaml")
            elif args.model == "qwen-30b-probe-v3-minilm-pixel":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_v3_minilm_pixel.yaml")
            elif args.model == "qwen-30b-probe-v3-minilm-pixel-ep1":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_v3_minilm_pixel_ep1.yaml")
            elif args.model == "qwen-30b-probe-v3-qwen-pca128-pixel":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_v3_qwen_pca128_pixel.yaml")
            elif args.model == "qwen-30b-probe-v3-qwen-pca128-pixel-ep1":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_v3_qwen_pca128_pixel_ep1.yaml")
            elif args.model == "qwen-30b-probe-v3-focal-minilm":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_v3_focal_minilm.yaml")
            elif args.model == "qwen-30b-probe-v3-focal-qwen":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_v3_focal_qwen.yaml")
            elif args.model == "qwen-30b-probe-crossdomain":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_crossdomain.yaml")
            elif args.model == "qwen-30b-probe-crossdomain-ep1":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_crossdomain_ep1.yaml")
            elif args.model == "qwen-30b-probe-hrbench4k-q3-ep1":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_hrbench4k_q3_ep1.yaml")
            elif args.model == "qwen-30b-probe-hrbench8k-q3-ep1":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_hrbench8k_q3_ep1.yaml")
            elif args.model == "qwen-30b-probe-cvbench-q3":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_cvbench_q3.yaml")
            elif args.model == "qwen-30b-probe-hrbench4k-q3":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_hrbench4k_q3.yaml")
            elif args.model == "qwen-30b-probe-hrbench8k-q3":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_hrbench8k_q3.yaml")
            elif args.model == "qwen-30b-probe-mme-q3":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_mme_q3.yaml")
            elif args.model == "qwen-30b-probe-vstar-q3":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_vstar_q3.yaml")
            elif args.model == "qwen-30b-probe-vstar":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_vstar.yaml")
            elif args.model == "qwen-235b-info-gain-lp":
                cfg_path = os.path.join(base_dir, "configs", "qwen235b_info_gain_lp.yaml")
            elif args.model == "qwen-30b-info-gain-lp":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_info_gain_lp.yaml")
            elif args.model == "qwen-30b-info-gain-lp-ep1":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_info_gain_lp_ep1.yaml")
            elif args.model == "qwen-30b-info-gain":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_info_gain.yaml")
            elif args.model == "qwen-30b-probe-gate":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_gate.yaml")
            elif args.model == "qwen-30b-probe-gate-tuned":
                cfg_path = os.path.join(base_dir, "configs", "qwen30b_probe_gate_tuned.yaml")
            elif args.model == "qwen-235b":
                cfg_path = os.path.join(base_dir, "configs", "qwen235b.yaml")
            elif args.model == "qwen-235b-belief":
                cfg_path = os.path.join(base_dir, "configs", "qwen235b_belief.yaml")
            elif args.model == "qwen-235b-voi-gate":
                cfg_path = os.path.join(base_dir, "configs", "qwen235b_voi_gate.yaml")
            elif args.model == "qwen-235b-probe":
                cfg_path = os.path.join(base_dir, "configs", "qwen235b_probe_235b.yaml")
            elif args.model == "qwen-235b-skillguard":
                cfg_path = os.path.join(base_dir, "configs", "qwen235b_skillguard.yaml")
            elif args.model == "internvl-8b":
                cfg_path = os.path.join(base_dir, "configs", "internvl8b.yaml")
            else:
                cfg_path = os.path.join(base_dir, "configs", "og_agent.yaml")
            logger.info("No config_path provided, using default for 'agent' mode: %s", cfg_path)
        elif args.mode == "aggregator":
            if args.model == "qwen-30b":
                cfg_path = os.path.join(base_dir, "configs", "aggregator.yaml")
            elif args.model == "qwen-32b":
                cfg_path = os.path.join(base_dir, "configs", "aggregator_t2.yaml")
            elif args.model == "qwen-235b":
                cfg_path = os.path.join(base_dir, "configs", "qwen235b.yaml")
            elif args.model == "qwen-235b-probe":
                cfg_path = os.path.join(base_dir, "configs", "qwen235b_probe_235b.yaml")
            elif args.model == "qwen-235b-skillguard":
                cfg_path = os.path.join(base_dir, "configs", "qwen235b_skillguard.yaml")
            logger.info(
                "No config_path provided, using default for 'aggregator' mode: %s", cfg_path
            )
        elif args.mode == "direct":
            # Direct mode still needs an LLM config. We can default to og_agent.yaml
            if args.model == "qwen-30b":
                cfg_path = os.path.join(base_dir, "configs", "og_agent.yaml")
            elif args.model == "qwen-235b":
                cfg_path = os.path.join(base_dir, "configs", "qwen235b.yaml")
            elif args.model == "gpt":
                cfg_path = os.path.join(base_dir, "configs", "gpt.yaml")
            elif args.model == "gemini":
                cfg_path = os.path.join(base_dir, "configs", "gemini.yaml")
            elif args.model == "claude":
                cfg_path = os.path.join(base_dir, "configs", "claude.yaml")
            elif args.model == "internvl-8b":
                cfg_path = os.path.join(base_dir, "configs", "internvl8b.yaml")
            else:
                raise ValueError("Unknown model '%s'.", args.model)

    if not os.path.exists(cfg_path):
        logger.error("Config file not found: %s", cfg_path)
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    # --- CLI overrides for VoI gate parameters ---
    if getattr(args, "alpha_table", None) is not None:
        OmegaConf.update(cfg, "workflow.nodes.action.parameters.voi_gate_path", args.alpha_table)
        logger.info("CLI override: voi_gate_path = %s", args.alpha_table)
    if getattr(args, "min_benefit", None) is not None:
        OmegaConf.update(cfg, "workflow.nodes.action.parameters.min_benefit", args.min_benefit)
        logger.info("CLI override: min_benefit = %s", args.min_benefit)
    if getattr(args, "template", None) is not None:
        OmegaConf.update(cfg, "workflow.nodes.agent.parameters.prompt_generator.parameters.template_path", args.template)
        logger.info("CLI override: template_path = %s", args.template)
    if getattr(args, "mcmc_iterations", None) is not None:
        OmegaConf.update(cfg, "agent_config.mcmc_iterations", args.mcmc_iterations)
        logger.info("CLI override: mcmc_iterations = %s", args.mcmc_iterations)
    if getattr(args, "mcmc_votes", None) is not None:
        OmegaConf.update(cfg, "agent_config.mcmc_votes", args.mcmc_votes)
        logger.info("CLI override: mcmc_votes = %s", args.mcmc_votes)
    invoke_config = get_invocation_config(cfg)
    assert isinstance(cfg, DictConfig)

    if args.mode in ("agent", "aggregator"):
        builder = GraphBuilder(cfg)
        agent_executor = builder.build()
        logger.info("Agent executor built for mode '%s'.", args.mode)
    elif args.mode == "direct":
        llm: BaseChatModel = chat_model_registry.get(cfg.llm.name, **cfg.llm.parameters)
    else:
        raise NotImplementedError(f"Mode '{args.mode}' not implemented in task runner.")

    logger.info("Loading dataset loader: %s", args.dataset)

    dataset_loader = get_dataset_loader(args.dataset)
    logger.info("Dataset loaded.")

    langfuse_client = get_client()

    # --- trajectory logger (one JSONL file per worker) ---
    trajectory_log_path = os.path.join(
        args.trajectory_dir, f"trajectories_worker_{worker_id}.jsonl"
    )
    traj_logger = TrajectoryLogger(trajectory_log_path)
    logger.info("Trajectory log: %s", trajectory_log_path)

    results = []
    for i, idx in enumerate(data_indices, 1):
        try:
            example = dataset_loader[idx]
            question = example["question"]
            correct_answer = example["correct_answer"]
            initial_image = example["image"]
            task_name = example["task_name"]
            sample_id = example["sample_id"]
        except Exception as e:
            logger.error("Failed to load dataset sample %d: %s", idx, e)
            continue  # Skip this sample

        turn_info = f"Sample {i}/{len(data_indices)} (Index {idx})"
        logger.info("Processing: %s", turn_info)
        logger.info("Task: %s", task_name)

        result_data = {
            "index": idx,
            "task": task_name,
            "question": question,
            "correct_answer": correct_answer,
            "agent_answer": "Error: Skipped (Agent run failed)",  # Default
        }

        # Initial an answer
        raw_response_content = None
        try:
            with langfuse_client.start_as_current_span(
                name=f"{sample_id}_{task_name}", input={"question": question}
            ) as current_span:
                # Propagate this metadata to all child calls
                # (LLM, tools) logged by the callback handler
                with propagate_attributes(tags=[task_name, sample_id, args.mode]):
                    trajectory_data: dict = {}
                    if args.mode in ("agent", "aggregator"):
                        agent_config_dict = {}
                        if "agent_config" in cfg:
                            # Convert OmegaConf object to a standard Python dict
                            agent_config_dict = OmegaConf.to_container(
                                cfg.agent_config, resolve=True
                            )
                        (
                            cvagent_answer,
                            raw_response_content,
                            tool_usage_stats,
                            trajectory_data,
                        ) = await run_agent_sample_with_retry(
                            question=question,
                            initial_image=initial_image,
                            agent_executor=agent_executor,
                            sample_id=sample_id,
                            invoke_config=invoke_config,
                            state_overrides={**agent_config_dict, '_ground_truth': correct_answer, 'task_name': task_name},
                            turn_info=turn_info,
                        )
                    elif args.mode == "direct":
                        cvagent_answer, raw_response_content = await run_direct_model_sample(
                            llm=llm,
                            question=question,
                            initial_image=initial_image,
                            sample_id=sample_id,
                            turn_info=turn_info,
                        )
                        tool_usage_stats = {}
                    else:
                        raise NotImplementedError
                result_data["agent_answer"] = cvagent_answer
                result_data["tool_usage"] = tool_usage_stats
                result_data["raw_response_content"] = raw_response_content
                # 1. Call grader
                grade = grade_answer(cvagent_answer, correct_answer)

                # 2. Log the grade.name ("CORRECT", "WRONG", "NO_ANSWER", "ERROR")
                current_span.score(
                    name="correctness",
                    value=grade.name,  # Use the enum name
                    data_type="CATEGORICAL",
                    comment=f"Agent: {cvagent_answer} | Correct: {correct_answer}",
                )

                # 3. Update the span's output
                if raw_response_content is not None:
                    current_span.update(
                        output={"agent_answer": cvagent_answer, "response": raw_response_content}
                    )
                else:
                    current_span.update(
                        output={
                            "agent_answer": cvagent_answer,
                        }
                    )

                current_span.update(
                    metadata={
                        "task_name": task_name,
                        "sample_id": sample_id,
                        "worker_id": worker_id,
                        "correct_answer": correct_answer,
                        "agent_answer": cvagent_answer,
                        "mode": args.mode,
                        "tool_usage_stats": json.dumps(tool_usage_stats),
                    },
                )

                # --- write trajectory JSONL record ---
                if trajectory_data:
                    traj_record = TrajectoryLogger.build_task_record(
                        task_id=sample_id,
                        image_url=trajectory_data.get("image_url", ""),
                        image_metadata={
                            "width": trajectory_data.get("img_width", 0),
                            "height": trajectory_data.get("img_height", 0),
                        },
                        query=question,
                        ground_truth=correct_answer,
                        final_answer=cvagent_answer,
                        is_correct=(grade == Grade.CORRECT),
                        trajectory_steps=trajectory_data.get("trajectory_steps", []),
                        total_input_tokens=trajectory_data.get("total_input_tokens", 0),
                        total_output_tokens=trajectory_data.get("total_output_tokens", 0),
                        task_name=task_name,
                        mode=args.mode,
                        episode_context=trajectory_data.get("episode_context", {}),
                        gate_log=trajectory_data.get("gate_log", []),
                    )
                    traj_logger.write(traj_record)

            logger.info("[%s] Question: %s", turn_info, question)
            logger.info("[%s] Correct Answer: %s", turn_info, correct_answer)
            logger.info("[%s] CVAgent Answer: %s\n%s", turn_info, cvagent_answer, "-" * 50)

        except Exception as e:
            logger.error("[%s] Critical failure: %s\n%s", turn_info, str(e), "-" * 50)
            result_data["agent_answer"] = f"Error: {e}"

        results.append(result_data)

    langfuse_client.flush()

    # --- Save Partial Results ---
    os.makedirs(args.results_dir, exist_ok=True)
    results_file_path = os.path.join(args.results_dir, f"results_worker_{worker_id}.json")
    with open(results_file_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Ray worker %d finished. Results saved to %s", worker_id, results_file_path)


# --- Ray Remote Wrapper ---
@ray.remote(num_cpus=1)
def main_ray_task(worker_id: int, data_indices: list, args: argparse.Namespace):
    """Ray remote function to run the async benchmark task."""
    asyncio.run(run_benchmark_task(worker_id, data_indices, args))


# --- merge_results ---


def merge_results(results_dir: str, final_file: str):
    """Merges all partial worker JSON results into a single file."""
    all_results = []
    main_logger = logging.getLogger()  # Get logger configured in __main__
    main_logger.info("Merging results from %s...", results_dir)
    try:
        for filename in os.listdir(results_dir):
            if filename.startswith("results_worker_") and filename.endswith(".json"):
                filepath = os.path.join(results_dir, filename)
                with open(filepath) as f:
                    all_results.extend(json.load(f))

        all_results.sort(key=lambda x: x["index"])
        final_path = os.path.join(results_dir, final_file)
        with open(final_path, "w") as f:
            json.dump(all_results, f, indent=2)
        main_logger.info("Successfully merged %d results into %s", len(all_results), final_path)
    except Exception as e:
        main_logger.error("Failed to merge results: %s", e)


# --- Main Entrypoint ---
if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent

    parser = argparse.ArgumentParser(description="Run Unified CV Agent Benchmark with Ray")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "mme",
            "cvbench",
            "vstar",
            "hrbench-4k",
            "hrbench-8k",
            "emb",
            "mme-og",
            "realworld",
            "probe-all",
            "probe-all-v2", "probe-all-v3", "probe-cub",
            "probe-gqa",
            "probe-textvqa",
            "probe-rsvlmqa", "probe-max", "probe-max-p1", "probe-max-p2",
        ],
        help="The name of the benchmark dataset to run.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["qwen-30b", "qwen-30b-ep2", "qwen-30b-logprobs", "qwen-30b-logprobs-ep2", "qwen-30b-skillguard", "qwen-30b-belief", "qwen-30b-voi-gate", "qwen-30b-info-gain", "qwen-30b-info-gain-lp", "qwen-30b-info-gain-lp-ep1", "qwen-235b-info-gain-lp", "qwen-30b-probe-gate", "qwen-30b-probe-gate-tuned", "qwen-30b-probe-cvbench", "qwen-30b-probe-hrbench4k", "qwen-30b-probe-hrbench8k", "qwen-30b-probe-mme", "qwen-30b-probe-vstar", "qwen-30b-probe-cvbench-q3", "qwen-30b-probe-hrbench4k-q3", "qwen-30b-probe-hrbench8k-q3", "qwen-30b-probe-mme-q3", "qwen-30b-probe-vstar-q3", "qwen-30b-probe-hrbench4k-q3-ep1", "qwen-30b-probe-hrbench8k-q3-ep1", "qwen-30b-probe-v3-skipsafe", "qwen-30b-probe-v3-skipsafe-qwen", "qwen-30b-probe-v3-skipsafe-qwen-c001", "qwen-30b-probe-v3-skipsafe-qwen-c001-ep1", "qwen-30b-probe-v3-skipsafe-qwen-ep1", "qwen-235b-probe-skipsafe-qwen", "qwen-235b-probe-skipsafe-minilm", "qwen-30b-probe-v3-skipsafe-ep1", "qwen-30b-probe-v3-minilm-pixel", "qwen-30b-probe-v3-minilm-pixel-ep1", "qwen-30b-probe-v3-qwen-pca128-pixel", "qwen-30b-probe-v3-qwen-pca128-pixel-ep1", "qwen-30b-probe-v3-focal-minilm", "qwen-30b-probe-v3-focal-qwen", "qwen-30b-probe-crossdomain", "qwen-30b-probe-cross-t30", "qwen-30b-probe-cross-t40", "qwen-30b-probe-cross-t50", "qwen-30b-probe-cross-t60", "qwen-30b-probe-cross-t70", "qwen-30b-probe-crossq3-t30", "qwen-30b-probe-crossq3-t40", "qwen-30b-probe-crossq3-t50", "qwen-30b-probe-crossq3-t60", "qwen-30b-probe-crossq3-t70", "qwen-30b-probe-cross-pv1", "qwen-30b-probe-cross-pv2", "qwen-30b-probe-cross-pv3", "qwen-30b-probe-qwenft-lora", "qwen-30b-probe-qwen-frozen", "qwen-30b-probe-v4", "qwen-30b-probe-v4-ep1", "qwen-30b-probe-v2-qwen-frozen", "qwen-30b-probe-v2-qwen-frozen-ep1", "qwen-30b-probe-qwen-frozen-ep1", "qwen-30b-probe-cross-pv3-ep1", "qwen-30b-probe-cross-pv3-t60", "qwen-30b-probe-cross-pv3-t60-ep1", "qwen-30b-probe-cross-pv3-t70", "qwen-30b-probe-cross-pv3-t70-ep1",  "qwen-30b-probe-crossdomain-q3", "qwen-30b-probe-crossdomain-q3-ep1", "qwen-30b-probe-crossdomain-ep1", "qwen-30b-info-gain-logprobs", "qwen-32b", "qwen-235b", "qwen-235b-probe", "qwen-235b-skillguard", "qwen-235b-belief", "qwen-235b-voi-gate", "internvl-8b", "qwen3-8b", "gpt", "gemini", "claude"],
        default="qwen-30b",
        help="The name of the model to run.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="agent",
        choices=["agent", "direct", "aggregator"],
        help="Run mode: 'agent' (ReAct-only), 'aggregator' (ReAct + Direct + Final),"
        " or 'direct' (base LLM).",
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Optional: Override default config path. 'agent' mode defaults to "
        "'react_agent_cv.yaml', 'aggregator' defaults to 'og_agent.yaml'.",
    )
    parser.add_argument("--tasks", type=str, default=None)

    # --- MME-specific args (optional) ---
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)

    # --- General args (Unchanged) ---
    parser.add_argument("--num_processes", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N samples.")
    parser.add_argument("--alpha-table", type=str, default=None,
                        help="Override alpha_table_path in VoI gate config.")
    parser.add_argument("--min-benefit", type=float, default=None,
                        help="Override min_benefit in VoI gate config.")
    parser.add_argument("--template", type=str, default=None,
                        help="Override prompt template path (e.g., planner_react_v1.j2).")
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated list of specific indices to run (e.g., '10,15,22').",
    )
    parser.add_argument("--mcmc-iterations", type=int, default=None,
                        help="Override MCMC max iterations (default: 6).")
    parser.add_argument("--mcmc-votes", type=int, default=None,
                        help="Override MCMC VLM vote count (default: 3).")

    # ---  Output directories now include dataset name ---
    args_temp, _ = parser.parse_known_args()  # Parse once to get dataset name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args_temp.mode == "direct":
        default_log_dir = os.path.join(
            base_dir, "logs", "direct_prompt", args_temp.dataset, timestamp
        )
        default_results_dir = os.path.join(
            base_dir, "results", "direct_prompt", args_temp.dataset, timestamp
        )
    else:
        # This now correctly creates separate dirs for 'agent' and 'aggregator'
        default_log_dir = os.path.join(
            base_dir, "logs", args_temp.mode, args_temp.dataset, timestamp
        )
        default_results_dir = os.path.join(
            base_dir, "results", args_temp.mode, args_temp.dataset, timestamp
        )

    model_alias = args_temp.model
    if args_temp.mode == "direct":
        default_log_dir = os.path.join(
            base_dir, "logs", model_alias, "direct_prompt", args_temp.dataset, timestamp
        )
        default_results_dir = os.path.join(
            base_dir, "results", model_alias, "direct_prompt", args_temp.dataset, timestamp
        )
    else:
        default_log_dir = os.path.join(
            base_dir, "logs", model_alias, args_temp.mode, args_temp.dataset, timestamp
        )
        default_results_dir = os.path.join(
            base_dir, "results", model_alias, args_temp.mode, args_temp.dataset, timestamp
        )

    # Add back log_dir, results_dir, and trajectory_dir with new defaults
    parser.add_argument("--log_dir", type=str, default=default_log_dir)
    parser.add_argument("--results_dir", type=str, default=default_results_dir)
    parser.add_argument(
        "--trajectory_dir",
        type=str,
        default=os.path.join(default_results_dir, "trajectories"),
        help="Directory to write per-worker JSONL trajectory files.",
    )

    parser.add_argument(
        "--samples_per_task",
        type=int,
        default=None,
        help="Randomly sample N items from each subtask. Overrides --limit.",
    )

    parser.add_argument(
        "--random_seed", type=int, default=0, help="Set a random seed for reproducible sampling."
    )

    # Re-parse arguments to get the correct log/results paths
    args = parser.parse_args()

    random.seed(args.random_seed)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    main_log_path = os.path.join(args.log_dir, "main.log")
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler(main_log_path), logging.StreamHandler()],
        format="[%(asctime)s] [MainThread] [%(levelname)s] %(message)s",
    )

    logger = logging.getLogger(__name__)  # Get logger for main thread

    ray.init(runtime_env={"working_dir": "/workspace/cv-agent", "excludes": [".venv", ".venv/**", "**/.venv/**", "results", "results/**", "logs", "logs/**", "data", "data/**", "reports/**", "__pycache__/**", "**/*.pyc", ".git/**", "pyproject.toml", "uv.lock", "*.ipynb"]})
    logger.info("Ray initialized. Log dir: %s, Results dir: %s", args.log_dir, args.results_dir)

    logger.info("Loading dataset loader: %s", args.dataset)
    all_indices = []
    try:
        # Load the loader once in the main thread to get its length
        dataset_loader = get_dataset_loader(args.dataset)
        all_indices = list(range(len(dataset_loader)))
        logger.info("Dataset '%s' loaded with %d samples.", args.dataset, len(all_indices))
    except Exception as e:
        logger.error("Failed to load dataset loader: %s", e)
        ray.shutdown()
        sys.exit(1)

    # --- TASK FILTERING LOGIC ---
    if args.tasks:
        # Lowercase the target tasks set: {'sensing', 'counting'}
        target_tasks = {t.strip().lower() for t in args.tasks.split(",")}
        logger.info("Filtering dataset to include tasks matching keywords: %s", target_tasks)

        filtered_indices = []
        for idx in all_indices:
            try:
                # 1. Get the sample's full task name (e.g., "Perception/Remote Sensing") and lowercase it
                sample_task_name = dataset_loader[idx].get("task_name", "Unknown").lower()

                # 2. Check for partial matching (SUBSTRING CHECK)
                # We check if *any* target keyword is a substring of the sample's task name.
                is_match = any(target_task in sample_task_name for target_task in target_tasks)

                if is_match:
                    filtered_indices.append(idx)
            except Exception as e:
                logger.warning(f"Could not check task name for index {idx}: {e}")

        # Replace all_indices with the filtered set
        all_indices = filtered_indices
        logger.info("Dataset filtered. Total samples now: %d", len(all_indices))

    if args.indices:
        try:
            all_indices = [int(i.strip()) for i in args.indices.split(",")]
            logger.info("Running on %d specific indices: %s", len(all_indices), all_indices)
        except Exception as e:
            logger.error("Invalid --indices format. Must be comma-separated numbers. Error: %s", e)
            sys.exit(1)
        # --- ADD THIS ENTIRE 'elif' BLOCK ---
    elif args.samples_per_task:
        logger.info(f"Building task map to sample {args.samples_per_task} items per task...")
        task_to_indices = defaultdict(list)
        for idx in all_indices:  # Assumes your full list is 'full_indices_list'
            try:
                task_name = dataset_loader[idx]["task_name"]
                task_to_indices[task_name].append(idx)
            except Exception as e:
                logger.warning(f"Could not load sample {idx} to get task_name: {e}")

        sampled_indices = []
        for _task_name, indices in task_to_indices.items():
            if len(indices) > args.samples_per_task:
                sampled_indices.extend(random.sample(indices, args.samples_per_task))
            else:
                sampled_indices.extend(indices)

        all_indices = sampled_indices
        logger.info(
            f"Created a new set of {len(all_indices)} total samples "
            f"from {len(task_to_indices)} tasks."
        )
    # --- END OF BLOCK ---
    elif args.limit:
        all_indices = all_indices[: args.limit]
        logger.info("Running on first %d samples (limit=%d).", len(all_indices), args.limit)
    else:
        logger.info("Running on all %d samples.", len(all_indices))

    num_processes = min(args.num_processes, len(all_indices))
    if num_processes == 0:
        logger.warning("No indices to process. Exiting.")
        ray.shutdown()
        sys.exit(0)

    chunk_size = (len(all_indices) + num_processes - 1) // num_processes
    parts = [all_indices[i : i + chunk_size] for i in range(0, len(all_indices), chunk_size)]
    logger.info("Dataset split into %d chunks for %d parallel workers.", len(parts), num_processes)

    results = []
    for i, part in enumerate(parts):
        if part:
            results.append(main_ray_task.remote(worker_id=i + 1, data_indices=part, args=args))

    try:
        ray.get(results)
        logger.info("All benchmark tasks completed.")
    except Exception as e:
        logger.error("Ray execution failed: %s", e)
    finally:
        ray.shutdown()
        logger.info("Ray shut down.")

    merge_results(args.results_dir, "final_benchmark_results.json")
    logger.info("Benchmark run finished.")

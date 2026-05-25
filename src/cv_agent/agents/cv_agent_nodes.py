"""Nodes for customized CV Agent."""

import asyncio
import copy
import json
import logging
import re
from typing import Any
import random


from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools.structured import StructuredTool

from cv_agent.core.registries import (
    Node,
    RoutingFunction,
    node_registry,
    prompt_generator_registry,
    routing_function_registry,
)
from cv_agent.utils.storage import download_image_to_pil

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trajectory helpers
# ---------------------------------------------------------------------------

def _extract_thinking_text(content) -> str:
    """Pull the **Thinking:** section out of the LLM response content.

    Handles three common formats produced by the planner prompt:
      1. ``**Thinking:**\\n<text>\\n---\\n``  (tool call follows)
      2. ``**Thinking:**\\n<text>\\n<answer>…</answer>``  (final answer)
      3. Raw text (fallback – return the whole string)
    """
    if not isinstance(content, str):
        return ""
    match = re.search(
        r"\*\*Thinking:\*\*\s*(.*?)(?:^---\s*$|\Z)",
        content,
        re.DOTALL | re.MULTILINE,
    )
    if match:
        return match.group(1).strip()
    return content.strip()


# ---------------------------------------------------------------------------

VISION_TOOL_NAMES = {
    "ocr",
    "detection",
    "segmentation",
    "cropping",
    "depth_estimation",
    "bbox_drawing",
    "bbox_center_labeling",
}

COORDINATE_INPUT_TOOLS = {
    "cropping",
    "bbox_drawing",
    "bbox_center_labeling",
}


def _try_parse_js_json(text: str) -> dict | None:
    """Try to parse JSON or JS-style object literal into a Python dict."""
    text = text.strip()
    # Attempt 1: standard JSON
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Attempt 2: fix unquoted keys (JS-style)
    # Match word chars before colon that aren't already quoted
    fixed = re.sub(r'(?<=[{,\s])(\w+)\s*:', r'"\1":', text)
    fixed = fixed.replace("'", '"')
    try:
        return json.loads(fixed)
    except (json.JSONDecodeError, ValueError):
        pass
    # Attempt 3: handle newlines, trailing commas
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)  # remove trailing commas
    try:
        return json.loads(fixed)
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _parse_text_tool_calls(content: str, available_tools: list) -> list[dict]:
    """Parse tool calls from model text output (fallback for models without structured tool calling).

    Handles formats like:
      <tool_call>functions.detection({image_url: "...", targets: ["car"]})</tool_call>
      <tool_call>detection({"image_url": "...", "targets": ["car"]})</tool_call>
      ```json\n{"name": "detection", ...}```
    """
    import uuid

    tool_names = {t.name for t in available_tools}
    calls = []

    # Pattern 1: <tool_call>...</tool_call>
    tc_matches = re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
    for tc_text in tc_matches:
        tc_text = tc_text.strip()
        # Extract function name: "functions.name({...})" or "name({...})"
        fn_match = re.match(r"(?:functions[.:])?(\w+)\s*\((.*)\)\s*$", tc_text, re.DOTALL)
        if not fn_match:
            continue
        fn_name = fn_match.group(1)
        args_text = fn_match.group(2).strip()

        if fn_name not in tool_names:
            continue

        # Try to parse args as JSON (might be JS-style without quotes on keys)
        args = _try_parse_js_json(args_text)
        if args is None:
            logger.warning("Could not parse tool call args: %s", args_text[:100])
            continue

        calls.append({
            "name": fn_name,
            "args": args,
            "id": f"textparse-{uuid.uuid4().hex[:8]}",
            "type": "tool_call",
        })

    # Pattern 2: ```json with tool name/args (if no <tool_call> found)
    if not calls:
        json_matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        for jm in json_matches:
            try:
                data = json.loads(jm)
            except json.JSONDecodeError:
                continue
            # Check if it looks like tool args with a known tool name mentioned nearby
            for tn in tool_names:
                if tn in content and any(k in data for k in ["image_url", "targets", "coordinates"]):
                    calls.append({
                        "name": tn,
                        "args": data,
                        "id": f"textparse-{uuid.uuid4().hex[:8]}",
                        "type": "tool_call",
                    })
                    break

    return calls


def generate_fake_url(state: dict, tool_name: str) -> str:
    """
    Generates a new, unique 'fake' URL for a generated image,
    based on the state's prefix and the tool that created it.
    """
    # 1. Get the prefix from the state (e.g., "cvbench_1")
    # This must be set in the initial state of your main script.
    prefix = state["prefix"]

    # 2. Create the base name for the new URL
    base_name = f"{prefix}_{tool_name}"  # e.g., "cvbench_1_cropping"

    key_prefix_to_check = f"http://{base_name}_"
    # 3. Count how many files with this base name already exist in the map
    url_map = state.get("url_map", {})
    count = 0
    for key in url_map:
        if key.startswith(key_prefix_to_check):
            count += 1

    # 4. The new number is the next one in the sequence
    new_number = count + 1

    # 5. Return the full fake URL
    return f"http://{base_name}_{new_number}.jpg"  # e.g., "cvbench_1_cropping_1.jpg"


@node_registry.register("planner_agent_node")
def get_planner_agent_node(
    chat_model: BaseChatModel,
    provided_tools: list[StructuredTool],
    prompt_generator: dict[str, Any],
    **kwargs,  # to be ignored
) -> Node:
    del kwargs
    generate_prompt = prompt_generator_registry.get(
        prompt_generator["name"], **prompt_generator.get("parameters", {})
    )
    chat_model_with_tools = chat_model.bind_tools(provided_tools)

    async def agent_node(state: dict) -> dict:
        print("--- AGENT NODE ---")
        messages = state.get("messages", [])

        current_turn = state["current_turn"]
        max_turns = state["max_turns"]
        if not messages:  # first round, need to initialize messages with question
            system_prompt = generate_prompt(state)
            messages.append(SystemMessage(content=system_prompt))

            original_image_url = state["original_figure_url"]
            user_question = state["question"]

            human_content = [
                {"type": "text", "text": user_question},
                {"type": "text", "text": f"The image url is {original_image_url}"},
                {"type": "image_url", "image_url": {"url": original_image_url}},
            ]

            # Append the single, multi-modal HumanMessage
            messages.append(HumanMessage(content=human_content))
            print(messages)

        else:
            reminder_text = (
                f"SYSTEM STATUS UPDATE: "
                f"You are currently on Turn {current_turn} out of {max_turns}.\n"
                f"If {current_turn} >= {max_turns}, "
                f"you MUST stop calling tools and "
                f"provide your best final answer immediately in the required format."
            )
            messages.append(HumanMessage(content=reminder_text))

        response = await chat_model_with_tools.ainvoke(messages)

        # Clean special tokens from response content (e.g., InternVL's <|channel|>, <|message|>, <|end|>)
        if response.content and isinstance(response.content, str):
            cleaned = re.sub(r"<\|channel\|>[^<]*<\|message\|>", "", response.content)
            cleaned = re.sub(r"<\|end\|>", "", cleaned).strip()
            if cleaned != response.content:
                response = AIMessage(
                    content=cleaned,
                    tool_calls=response.tool_calls,
                    additional_kwargs=response.additional_kwargs,
                )

        # Fallback: parse tool calls from text if model doesn't produce structured ones
        if not response.tool_calls and response.content:
            parsed_calls = _parse_text_tool_calls(response.content, provided_tools)
            if parsed_calls:
                response = AIMessage(
                    content=response.content,
                    tool_calls=parsed_calls,
                    additional_kwargs=response.additional_kwargs,
                )
                logger.info("Parsed %d tool call(s) from text content", len(parsed_calls))

        print(response)
        # manually update the state
        messages.append(response)
        state.update({"messages": messages})
        # current turn increment
        state["current_turn"] = current_turn + 1

        # --- trajectory: capture thinking text for the tool executor to pick up ---
        _response_text = response.content if isinstance(response.content, str) else ""
        state["pending_thinking"] = _extract_thinking_text(_response_text)

        # --- trajectory: capture thinking logprobs if available ---
        _resp_meta = getattr(response, "response_metadata", {}) or {}
        _logprobs_data = _resp_meta.get("logprobs", {})
        if _logprobs_data and _logprobs_data.get("content"):
            _token_logprobs = [t["logprob"] for t in _logprobs_data["content"]]
            _top_logprobs = [
                {tt["token"]: tt["logprob"] for tt in t.get("top_logprobs", [])}
                for t in _logprobs_data["content"]
            ]
            state["pending_thinking_logprobs"] = _token_logprobs
            state["pending_thinking_top_logprobs"] = _top_logprobs
            # Sidecar: write logprobs to file (LangGraph strips state/metadata)
            import os as _os
            _sidecar = _os.environ.get("LOGPROBS_SIDECAR")
            if _sidecar:
                import json as _json
                _rec = {
                    "question_key": state.get("question", "")[:80],
                    "url": str(state.get("original_figure_url", "")),
                    "turn": current_turn,
                    "token_logprobs": _token_logprobs,
                    "top_logprobs": _top_logprobs[:30],
                }
                with open(_sidecar, "a") as _f:
                    _f.write(_json.dumps(_rec, ensure_ascii=False) + "\n")

        else:
            state["pending_thinking_logprobs"] = []
            state["pending_thinking_top_logprobs"] = []

        # --- belief annotation: parse <belief> block ---
        state["pending_belief"] = {}

        # --- belief annotation: context ξ (first turn only) ---
        if current_turn == 1 and state.get("enable_belief_annotation", False):
            try:
                ctx_prompt = (
                    "Analyze this image and question. Respond ONLY with JSON "
                    "(no markdown fences):\n"
                    '{"query_type": "<counting|spatial_relation|depth_distance|'
                    'attribute|reading_ocr|chart_diagram>",'
                    ' "domain": "<natural_scene|remote_sensing|monitoring|'
                    'autonomous_driving|document|indoor>",'
                    ' "scene_complexity": "<sparse|moderate|cluttered>"}'
                )
                ctx_human_content = [
                    {"type": "text", "text": f"Question: {state['question']}"},
                    {"type": "text", "text": ctx_prompt},
                    {"type": "image_url", "image_url": {"url": state["original_figure_url"]}},
                ]
                ctx_response = await chat_model.ainvoke(
                    [HumanMessage(content=ctx_human_content)]
                )
                # Extract JSON from response
                ctx_text = ctx_response.content if isinstance(ctx_response.content, str) else ""
                ctx_json_match = re.search(r"\{.*\}", ctx_text, re.DOTALL)
                if ctx_json_match:
                    episode_ctx = json.loads(ctx_json_match.group(0))
                else:
                    episode_ctx = {}
                # Add image resolution from state
                dims = state.get("image_dimensions", {})
                original_url = state.get("original_figure_url", "")
                if original_url in dims:
                    w, h = dims[original_url]
                    episode_ctx["resolution"] = f"{w}x{h}"
                state["episode_context"] = episode_ctx
                logger.info("Belief annotation: episode_context=%s", episode_ctx)
            except Exception as e:
                logger.warning("Belief annotation: context annotation failed: %s", e)
                state["episode_context"] = {}

        # --- trajectory: final answer → append terminal step ---
        if "<answer>" in (_response_text or "").lower():
            state.setdefault("trajectory_steps", []).append({
                "step": current_turn,
                "thinking": state["pending_thinking"],
                "belief": state.get("pending_belief", {}),
                "tool_name": "__final_answer__",
                "tool_params": {},
                "input_image_url": "",
                "tool_output": {},
                "output_image_url": None,
            })

        # --- trajectory: accumulate token counts ---
        usage = getattr(response, "usage_metadata", None) or {}
        state["total_input_tokens"] = (
            state.get("total_input_tokens", 0) + usage.get("input_tokens", 0)
        )
        state["total_output_tokens"] = (
            state.get("total_output_tokens", 0) + usage.get("output_tokens", 0)
        )

        return state

    return agent_node


async def _get_image_dimensions(state: dict, image_url: str) -> tuple[int, int]:
    """
    Gets image dimensions, reading from state first and downloading as a fallback.
    Saves new dimensions back to the state.
    """
    # 1. Check if dimensions are already in the state
    if "image_dimensions" not in state:
        state["image_dimensions"] = {}

    if image_url in state["image_dimensions"]:
        logger.info("Got dimensions for %s... from state cache.", image_url[:50])
        return state["image_dimensions"][image_url]

    # 2. If not, download the image to get them
    logger.info("Dimensions for %s... not in state, downloading.", image_url[:50])
    try:
        pil_image = await download_image_to_pil(image_url)
        width, height = pil_image.size

        # 3. Save new dimensions back to the state for future use
        state["image_dimensions"][image_url] = (width, height)

        return width, height
    except Exception as e:
        logger.error("Failed to _get_image_dimensions for %s: %s", image_url, e)
        raise ValueError(f"Could not determine dimensions for image: {image_url}") from e


async def _convert_to_absolute_coords(
    state: dict, tool_args: dict, image_url: str, tool_name: str
) -> dict:
    """
    INBOUND: Converts relative [0, 1000] coords from Agent
    to absolute pixel coords for the Tool.
    """
    try:
        img_width, img_height = await _get_image_dimensions(state, image_url)
        if img_width == 0 or img_height == 0:
            raise ValueError(f"Invalid dimensions for image: {image_url}")

        scale_x = img_width / 1000.0
        scale_y = img_height / 1000.0

        if tool_name == "cropping" and "coordinates" in tool_args:
            rel_coords = tool_args["coordinates"]
            if len(rel_coords) == 4:
                tool_args["coordinates"] = [
                    int(rel_coords[0] * scale_x),
                    int(rel_coords[1] * scale_y),
                    int(rel_coords[2] * scale_x),
                    int(rel_coords[3] * scale_y),
                ]

        elif tool_name == "bbox_drawing" and "boxes" in tool_args:
            for box_info in tool_args.get("boxes", []):
                rel_bbox = box_info.get("bbox")
                if rel_bbox and len(rel_bbox) == 4:
                    box_info["bbox"] = [
                        int(rel_bbox[0] * scale_x),
                        int(rel_bbox[1] * scale_y),
                        int(rel_bbox[2] * scale_x),
                        int(rel_bbox[3] * scale_y),
                    ]

        elif tool_name == "bbox_center_labeling" and "bboxes" in tool_args:
            abs_bboxes = []
            for rel_bbox in tool_args.get("bboxes", []):
                if len(rel_bbox) == 4:
                    abs_bboxes.append(
                        [
                            int(rel_bbox[0] * scale_x),
                            int(rel_bbox[1] * scale_y),
                            int(rel_bbox[2] * scale_x),
                            int(rel_bbox[3] * scale_y),
                        ]
                    )
            tool_args["bboxes"] = abs_bboxes

        return tool_args
    except Exception as e:
        logger.error("Failed INBOUND coordinate conversion for %s: %s", tool_name, e)
        raise


async def _convert_to_relative_coords(state: dict, tool_output_data: dict, image_url: str) -> dict:
    """
    OUTBOUND: Converts absolute pixel coords from Detection Tool
    to relative [0, 1000] coords for the Agent.
    """
    try:
        img_width, img_height = await _get_image_dimensions(state, image_url)
        if img_width == 0 or img_height == 0:
            raise ValueError(f"Invalid dimensions for image: {image_url}")

        scale_x = 1000.0 / img_width
        scale_y = 1000.0 / img_height

        new_output_text = []

        if "detections" in tool_output_data and tool_output_data["detections"]:
            for detection in tool_output_data["detections"]:
                abs_bbox = detection.get("bbox")
                if abs_bbox and len(abs_bbox) == 4:
                    # 1. Translate from absolute to relative [0, 1000]
                    rel_bbox = [
                        int(abs_bbox[0] * scale_x),
                        int(abs_bbox[1] * scale_y),
                        int(abs_bbox[2] * scale_x),
                        int(abs_bbox[3] * scale_y),
                    ]
                    detection["bbox"] = rel_bbox

                    # 2. Re-generate the output_text string
                    phrase = detection.get("phrase", "object")
                    confidence = detection.get("confidence", 0.0)
                    new_text = (
                        f"Detected {phrase} with confidence {confidence:.3f} "
                        f"at location [{rel_bbox[0]}, {rel_bbox[1]}, {rel_bbox[2]}, {rel_bbox[3]}]"
                    )
                    new_output_text.append(new_text)

        if new_output_text:
            tool_output_data["output_text"] = new_output_text
            logger.info("Normalized 'output_text' for agent.")
        return tool_output_data
    except Exception as e:
        logger.error("Failed OUTBOUND coordinate conversion for detection: %s", e)
        raise

async def _evaluate_crop_candidates_mcmc(
    tool: StructuredTool,
    base_args: dict,
    chat_model: BaseChatModel,
    state: dict,
    image_url: str,
) -> Any:
    """
    基于 MCMC (马尔科夫链蒙特卡洛) 的自适应采样策略。
    根据当前框的大小动态调整移动步长，通过迭代寻找最佳回答区域。
    """
    logger.info("Starting MCMC Sampling with Adaptive Step Size...")

    try:
        img_width, img_height = await _get_image_dimensions(state, image_url)
        # 初始状态 X_0: Agent 给出的原始坐标
        current_coords = list(base_args["coordinates"])
    except Exception as e:
        logger.warning(f"MCMC pre-check failed: {e}")
        return await tool.ainvoke(base_args)

    user_question = state.get("question", "")

    # --- MCMC 参数配置 (可通过 state 覆盖) ---
    MAX_ITERATIONS = state.get("mcmc_iterations", 6)
    ETA = 0.15  # 扰动系数：移动标准差为当前框宽高的 15%
    NUM_VOTES = state.get("mcmc_votes", 3)
    logger.info(f"MCMC config: iterations={MAX_ITERATIONS}, votes={NUM_VOTES}, eta={ETA}")

    best_observation = None
    best_score = -1.0

    # 当前链的状态
    current_score = -1.0
    current_obs = None

    # --- 内部函数：获取 VLM 评分 (Probability) ---
    async def get_vlm_score(crop_url):
        system_msg = """# Task
        Evaluate whether a cropped region contains sufficient information to answer a question.

        # Context
        You will be shown two images:
        1. The original image (Image 1)
        2. A candidate cropped region (Image 2)

        # Instructions
        1. Analyze the correlation: What's the relationship between these images? How does the candidate cropped region relate to the original question?
        2. Identify mismatch: If the candidate cropped region is on a different part of the original image from where the question asks, this region contains no information and you should answer with "No".
        3. Be confident of your choice: Trust your reasoning and perception, based on which you can always give a "Yes" or "No" to whether this cropped region helps answer the question.

        # Constraints
        - MUST NOT answer anything other than "Yes" and "No" inside the answer tag.
        - MUST use the thinking section to reason about the relationships between images and the question, and give a thorough analysis.
        - ALWAYS output reasoning under `**Thinking:**` and "Yes" or "No" within `<answer>`.

        # Output Format
        **Thinking:**
        [Your step-by-step reasoning here]
        **Answer:**
        <answer>Yes or No</answer>
        """

        human_msg = [
            {"type": "text", "text": f'Original Question: "{user_question}"'},
            {"type": "text", "text": "Image 1: Original Image"},
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": "Image 2: Candidate Crop"},
            {"type": "image_url", "image_url": {"url": crop_url}},
        ]

        messages = [SystemMessage(content=system_msg), HumanMessage(content=human_msg)]

        try:
            # 并行投票获取概率
            vote_tasks = [chat_model.ainvoke(messages) for _ in range(NUM_VOTES)]
            results = await asyncio.gather(*vote_tasks, return_exceptions=True)

            yes_count = 0
            valid_votes = 0
            for res in results:
                if isinstance(res, Exception):
                    continue
                valid_votes += 1
                # 使用正则提取 <answer>Yes</answer>
                if re.search(r"<answer>\s*Yes\s*</answer>", res.content, re.IGNORECASE):
                    yes_count += 1

            return (yes_count / valid_votes) if valid_votes > 0 else 0.0
        except Exception as e:
            logger.error(f"VLM scoring error: {e}")
            return 0.0

    # --- 开始 MCMC 迭代 ---
    for i in range(MAX_ITERATIONS):
        # 1. 提议分布 (Proposal): 基于当前比例生成新坐标
        x1, y1, x2, y2 = current_coords
        w_crop = x2 - x1
        h_crop = y2 - y1

        # 计算自适应步长（当前框宽高的 ETA 倍）
        sigma_x = w_crop * ETA
        sigma_y = h_crop * ETA

        # 随机扰动：平移 + 轻微缩放
        dx = random.gauss(0, sigma_x)
        dy = random.gauss(0, sigma_y)
        dw = random.gauss(0, w_crop * 0.05)  # 宽高 5% 抖动
        dh = random.gauss(0, h_crop * 0.05)

        proposal_coords = [
            int(max(0, min(img_width - 20, x1 + dx - dw / 2))),
            int(max(0, min(img_height - 20, y1 + dy - dh / 2))),
            int(max(x1 + 20, min(img_width, x2 + dx + dw / 2))),
            int(max(y1 + 20, min(img_height, y2 + dy + dh / 2))),
        ]

        logger.info(
            f"MCMC Iter {i}: Proposing {proposal_coords} (Step size based on {w_crop}x{h_crop})"
        )

        # 2. 调用工具生成预览图
        temp_args = copy.deepcopy(base_args)
        temp_args["coordinates"] = proposal_coords

        try:
            obs = await asyncio.wait_for(tool.ainvoke(temp_args), timeout=15)
            obs_data = json.loads(obs.content[0].text)
            crop_url = obs_data.get("output_image")

            if not crop_url:
                continue

            # 3. 计算得分 (Acceptance Ratio)
            proposal_score = await get_vlm_score(crop_url)

            # 初始分评估（仅针对第一步）
            if current_score < 0:
                # 如果是第一步，先给原 base_coords 算个分做基准
                # 但为了节省 Token，我们直接假设当前 proposal 是第一个有效状态
                current_score = proposal_score
                current_obs = obs
                current_coords = proposal_coords
            else:
                # Metropolis-Hastings 接受决策
                # 如果新分更高，或者以一定概率（0.1）接受较差的结果以跳出局部最优
                if proposal_score >= current_score or random.random() < 0.1:
                    current_score = proposal_score
                    current_obs = obs
                    current_coords = proposal_coords
                    logger.info(f"Accepted move. New score: {current_score}")

            # 记录历史最优
            if current_score > best_score:
                best_score = current_score
                best_observation = current_obs

            # 提前停止条件：如果得分已经是 1.0 (全票 Yes)，直接返回
            if best_score >= 1.0:
                logger.info(f"Found optimal crop at iteration {i}. Score: 1.0")
                return best_observation

        except Exception as e:
            logger.error(f"MCMC iteration {i} error: {e}")
            continue

    # 如果迭代完成，返回过程中表现最好的观察结果
    if best_observation:
        logger.info(f"MCMC finished. Best score: {best_score}")
        return best_observation

    # 兜底：返回原始参数的执行结果
    return await tool.ainvoke(base_args)

async def _evaluate_crop_candidates(
    tool: StructuredTool,
    base_args: dict,
    chat_model: BaseChatModel,
    state: dict,
    image_url: str,
) -> Any:
    """
    Implements Predictive Sampling using the user-provided CoT Prompt.
    """
    # 1. Get Image Dimensions & Base Coordinates
    try:
        img_width, img_height = await _get_image_dimensions(state, image_url)
        base_coords = base_args["coordinates"]
    except Exception as e:
        logger.warning(f"Skipping sampling due to error: {e}")
        return await tool.ainvoke(base_args)

    user_question = state.get("question", "")

    # Helper: Scale Box
    def get_candidate_box(coords, scale_factor):
        x1, y1, x2, y2 = coords
        w = x2 - x1
        h = y2 - y1
        cx, cy = x1 + w / 2, y1 + h / 2
        new_w = w * scale_factor
        new_h = h * scale_factor
        nx1 = int(max(0, cx - new_w / 2))
        ny1 = int(max(0, cy - new_h / 2))
        nx2 = int(min(img_width, cx + new_w / 2))
        ny2 = int(min(img_height, cy + new_h / 2))
        return [nx1, ny1, nx2, ny2]

    # Generate Candidates
    candidates = [
        {"name": "original", "coords": base_coords},
        {"name": "large", "coords": get_candidate_box(base_coords, 1.3)},
        {"name": "small", "coords": get_candidate_box(base_coords, 0.8)},
    ]

    best_observation = None
    best_score = -1
    best_candidate_name = "original"

    for cand in candidates:
        name = cand["name"]
        coords = cand["coords"]

        temp_args = copy.deepcopy(base_args)
        temp_args["coordinates"] = coords

        try:
            # 获取裁剪图
            obs = await asyncio.wait_for(tool.ainvoke(temp_args), timeout=15)

            if not obs.content or obs.content[0].type != "text":
                continue

            content_json = obs.content[0].text
            if isinstance(content_json, str):
                data = json.loads(content_json)
            else:
                data = content_json

            crop_url = data.get("output_image")
            if not crop_url:
                continue

            # ==========================================
            #   Prompt 注入区域 (User Provided)
            # ==========================================


            system_msg = """# Task
Evaluate whether a cropped region contains sufficient information to answer a question.

# Context
You will be shown two images:
1. The original image (Image 1)
2. A candidate cropped region (Image 2)

# Instructions
1. Analyze the correlation: What's the relationship between these images? How does the candidate cropped region relate to the original question?
2. Identify mismatch: If the candidate cropped region is on a different part of the original image from where the question asks, this region contains no information and you should answer with "No".
3. Be confident of your choice: Trust your reasoning and perception, based on which you can always give a "Yes" or "No" to whether this cropped region helps answer the question.

# Constraints
- MUST NOT answer anything other than "Yes" and "No" inside the answer tag.
- MUST use the thinking section to reason about the relationships between images and the question, and give a thorough analysis.
- ALWAYS output reasoning under `**Thinking:**` and "Yes" or "No" within `<answer>`.

# Output Format
**Thinking:**
[Your step-by-step reasoning here]
**Answer:**
<answer>Yes or No</answer>
"""

            human_msg = [
                {
                    "type": "text",
                    "text": (
                        f'Original Question: "{user_question}"\n\n'
                        "Please evaluate the candidates below:\n"
                    ),
                },
                {"type": "text", "text": "Image 1: The Original Image"},
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "Image 2: The Candidate Cropped Region"},
                {"type": "image_url", "image_url": {"url": crop_url}},
            ]

            messages = [SystemMessage(content=system_msg), HumanMessage(content=human_msg)]

            # 因为需要输出 Thinking，Token 数会增加，Timeout 设为 25-30s
            NUM_VOTES = 3
            vote_tasks = [chat_model.ainvoke(messages) for _ in range(NUM_VOTES)]

            try:
                vote_results = await asyncio.wait_for(
                    asyncio.gather(*vote_tasks, return_exceptions=True), timeout=25
                )
            except asyncio.TimeoutError:
                # logger.warning(f"VLM voting timed out for {name}")
                continue

            yes_count = 0
            valid_votes = 0

            for res in vote_results:
                if isinstance(res, Exception):
                    continue
                valid_votes += 1
                content = res.content

                # print(f"[{name} Reasoning]: {content[:200]}...") # Debug: 看看它的推理过程

                # ==========================================
                #   正则解析 (Robust Parsing)
                # ==========================================
                # 因为模型输出包含 Thinking，我们必须使用正则提取 <answer>
                match = re.search(r"<answer>\s*(Yes|No|yes|no)\s*</answer>", content, re.IGNORECASE)

                if match:
                    answer_val = match.group(1).lower()
                    if answer_val == "yes":
                        yes_count += 1
                else:
                    # Fallback: 如果模型忘了写 Tag，但 Thinking 里最后说是 Yes
                    # 风险较高，仅作保底，或者直接视为 No
                    if "**Answer:**\nYes" in content or "**Answer:** Yes" in content:
                        yes_count += 1

            score = (yes_count / valid_votes) if valid_votes > 0 else 0

            # logger.info(f"Candidate '{name}' Score: {score:.2f}")

            if score > best_score:
                best_score = score
                best_observation = obs
                best_candidate_name = name

            # 如果 Original 已经完美 (3/3 Yes)，提前结束
            if name == "original" and score == 1.0:
                logger.info("Original crop is perfect. Stopping search.")
                return best_observation

        except Exception as e:
            logger.error(f"Error evaluating candidate '{name}': {e}")
            continue

    if best_observation:
        logger.info(f"Selected Best Candidate: '{best_candidate_name}'")
        return best_observation

    logger.warning("Sampling failed. Running original args.")
    return await asyncio.wait_for(tool.ainvoke(base_args), timeout=15)


@node_registry.register("cv_tool_executor")
def get_cv_tool_executor(
    executable_tools: dict[str, StructuredTool], chat_model: BaseChatModel, **kwargs
) -> Node:
    """
    This is the node that executes the tools.
    It acts as a two-way translation layer for coordinates.
    """
    # Extract configuration for sampling
    enable_sampling = kwargs.get("enable_sampling", False)
    sampling_strategy = kwargs.get("sampling_strategy", "naive")
    # Coordinate system: if True, skip [0,1000] ↔ pixel conversion (model uses absolute coords)
    use_absolute_coords = kwargs.get("use_absolute_coords", False)
    # Extract SkillGuard configuration
    enable_skillguard = kwargs.get("enable_skillguard", False)
    # VoI gate removed; probe gate is the method
    voi_gate = None
    # Extract Probe Gate configuration
    enable_probe_gate = kwargs.get("enable_probe_gate", False)
    probe_gate = None
    if enable_probe_gate:
        probe_gate_path = kwargs.get("probe_gate_model_path", "")
        probe_gate_threshold = kwargs.get("probe_gate_threshold", 0.5)
        from cv_agent.skillguard.probe_gate import ProbeGate
        probe_gate = ProbeGate(probe_gate_path, threshold=probe_gate_threshold,
            force_answer_message=kwargs.get("force_answer_message"),
            skip_message=kwargs.get("skip_message"))
        logger.info("Probe Gate enabled: model=%s, threshold=%.2f", probe_gate_path, probe_gate_threshold)
    # Info gain collection mode
    collect_info_gain = kwargs.get("collect_info_gain", False)
    info_gain_endpoint = kwargs.get("info_gain_endpoint", "")
    info_gain_model = kwargs.get("info_gain_model", "")

    # If chat_model was passed in kwargs (by registry), use it.
    if chat_model is None and "chat_model" in kwargs:
        chat_model = kwargs["chat_model"]

    async def tool_executor(state: dict) -> dict:
        print("--- TOOL EXECUTOR (Vision-Enabled) ---")

        last_msg = state["messages"][-1]
        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            return state

        tool_msgs = []

        tool_usage = state["tool_usage"]

        url_map = state["url_map"]

        for tool_call in last_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1

            if tool_name not in executable_tools:
                logger.error(f"Agent hallucinated non-existent tool: {tool_name}")
                tool_msgs.append(
                    ToolMessage(
                        content=json.dumps(
                            {
                                "status": "error",
                                "message": f"Tool '{tool_name}' is not a valid tool name.",
                            }
                        ),
                        tool_call_id=tool_call["id"],
                    )
                )
                continue  # Skip to next tool call or message processing

            tool = executable_tools[tool_call["name"]]
            tool_args = copy.deepcopy(tool_call["args"])

            # --- VoI Gate: decide whether to execute this tool call ---
            gate_info: dict | None = None
            if voi_gate is not None:
                from cv_agent.skillguard.voi_gate import SKIP_MESSAGE, FORCE_ANSWER_MESSAGE, MAX_CONSECUTIVE_SKIPS
                execute, gate_info = voi_gate.should_execute(
                    tool_name,
                    state.get("episode_context", {}),
                    state.get("pending_belief", {}),
                )
                state.setdefault("gate_log", []).append({
                    "step": state.get("current_turn", 0) - 1,
                    "tool_name": tool_name,
                    **gate_info,
                })
                if not execute:
                    # Track consecutive skips
                    consec = state.get("_consecutive_skips", 0) + 1
                    state["_consecutive_skips"] = consec
                    logger.info("VoI Gate: skipping %s (reason=%s, consecutive=%d)", tool_name, gate_info.get("reason"), consec)
                    _step_record = {
                        "step": state.get("current_turn", 0) - 1,
                        "thinking": state.get("pending_thinking", ""),
                        "belief": state.get("pending_belief", {}),
                        "tool_name": tool_name,
                        "tool_params": tool_call["args"],
                        "input_image_url": "",
                        "tool_output": {"status": "skipped", "reason": gate_info.get("reason", "voi_negative")},
                        "output_image_url": None,
                        "gate_decision": gate_info,
                    }
                    state.setdefault("trajectory_steps", []).append(_step_record)
                    tool_msgs.append(ToolMessage(content=SKIP_MESSAGE, tool_call_id=tool_call["id"]))
                    # Force answer after too many consecutive skips
                    if consec >= MAX_CONSECUTIVE_SKIPS:
                        tool_msgs.append(HumanMessage(content=FORCE_ANSWER_MESSAGE))
                        logger.info("VoI Gate: forcing answer after %d consecutive skips", consec)
                    continue
                else:
                    # Reset consecutive skip counter on execute
                    state["_consecutive_skips"] = 0

            # --- Probe Gate: decide whether to execute this tool call ---
            if probe_gate is not None:
                from cv_agent.skillguard.probe_gate import (
                    SKIP_MESSAGE as PG_SKIP_MESSAGE,
                    get_skip_message as pg_get_skip_message,
                    FORCE_ANSWER_MESSAGE as PG_FORCE_ANSWER_MESSAGE,
                    MAX_CONSECUTIVE_SKIPS as PG_MAX_CONSECUTIVE_SKIPS,
                )
                p_execute, p_gate_info = probe_gate.should_execute(
                    tool_name,
                    tool_call["args"],
                    state,
                )
                state.setdefault("gate_log", []).append({
                    "step": state.get("current_turn", 0) - 1,
                    "tool_name": tool_name,
                    **p_gate_info,
                })
                if not p_execute:
                    consec = state.get("_consecutive_skips", 0) + 1
                    state["_consecutive_skips"] = consec
                    logger.info("Probe Gate: skipping %s (p=%.3f, consecutive=%d)",
                                tool_name, p_gate_info.get("p_tool_useful", 0), consec)
                    _step_record = {
                        "step": state.get("current_turn", 0) - 1,
                        "thinking": state.get("pending_thinking", ""),
                        "belief": state.get("pending_belief", {}),
                        "tool_name": tool_name,
                        "tool_params": tool_call["args"],
                        "input_image_url": "",
                        "tool_output": {"status": "skipped", "reason": p_gate_info.get("reason", "probe_not_useful")},
                        "output_image_url": None,
                        "gate_decision": p_gate_info,
                    }
                    state.setdefault("trajectory_steps", []).append(_step_record)
                    tool_msgs.append(ToolMessage(content=probe_gate._override_skip or pg_get_skip_message(tool_name), tool_call_id=tool_call["id"]))
                    if consec >= PG_MAX_CONSECUTIVE_SKIPS:
                        tool_msgs.append(HumanMessage(content=probe_gate._override_force or PG_FORCE_ANSWER_MESSAGE))
                        logger.info("Probe Gate: forcing answer after %d consecutive skips", consec)
                    continue
                else:
                    state["_consecutive_skips"] = 0

            # --- Info gain collection: logprobs BEFORE tool ---
            _info_gain_logprobs_before = None
            if collect_info_gain and info_gain_endpoint:
                try:
                    from cv_agent.skillguard.info_gain import get_answer_logprobs, messages_to_api_format
                    api_msgs = messages_to_api_format(state["messages"])
                    _info_gain_logprobs_before = get_answer_logprobs(
                        api_msgs, info_gain_endpoint, info_gain_model
                    )
                    logger.info("Info gain: logprobs_before collected for %s", tool_name)
                except Exception as e:
                    logger.warning("Info gain: failed to collect logprobs_before: %s", e)

            human_msg_to_add = None

            # --- trajectory: initialise per-call tracking vars ---
            _traj_data: dict = {}
            _traj_input_image_url: str = ""
            _traj_output_image_url: str | None = None

            try:
                # 1. Find the image URL this tool is supposed to act on
                # If the agent doesn't provide the 'image_url' argument,
                # it automatically defaults to "original_figure_url".
                image_key_to_check = tool_args.get("image_url", "original_figure_url")

                # 2. Check if this key exists in the map.
                # The 'url_map' contains "original_figure_url" AND all fake URLs.

                if image_key_to_check in url_map:
                    # It's a valid key. Use the real URL.
                    image_url_to_use = url_map[image_key_to_check]

                else:
                    # 3. This is the 'else' block you wanted.
                    # The key is not 'original_figure_url' and not a valid fake URL.
                    # We raise an error to send back to the agent.
                    logger.error(f"Agent provided a non-existent image key: {image_key_to_check}")
                    raise ValueError(
                        f"The image key '{image_key_to_check}' does not exist in the url_map. "
                        f"Available keys are: {list(url_map.keys())}"
                    )

                # Replace to real url direct to the server
                tool_args["image_url"] = image_url_to_use
                _traj_input_image_url = image_url_to_use  # trajectory
                # 2. INBOUND Translation: (Agent -> Tool)
                if tool_name in COORDINATE_INPUT_TOOLS and not use_absolute_coords:
                    logger.info("Converting INBOUND coords for %s...", tool_name)
                    tool_args = await _convert_to_absolute_coords(
                        state, tool_args, image_url_to_use, tool_name
                    )
                    logger.info("Converted args: %s", tool_args)

                # DEBUG TRACE
                print(f"[DEBUG] Checking Sampling: enable={enable_sampling}, tool={tool_name}, has_model={chat_model is not None}")
                
                # 3. Run the tool (With potential intervention)
                if enable_sampling and tool_name == "cropping" and chat_model:
                    print("[DEBUG] Entering Sampling Block")
                    if sampling_strategy == "naive":
                        observation = await _evaluate_crop_candidates(
                            tool, tool_args, chat_model, state, image_url_to_use
                        )
                    elif sampling_strategy == "mcmc":
                        observation = await _evaluate_crop_candidates_mcmc(
                            tool, tool_args, chat_model, state, image_url_to_use)
                    else:
                        raise Exception("Sampling strategy not recognized")
                else:
                    print(f"[DEBUG] Entering Standard Block. Args: {tool_args}")
                    logger.info("Calling Tool with ARGS: %s", tool_args)
                    observation = await asyncio.wait_for(tool.ainvoke(tool_args), timeout=30)

                logger.debug("Raw tool output: %s", observation)

                # 4. Parse the CallToolResult object (which might be string or dict)
                if not observation.content or observation.content[0].type != "text":
                    raise Exception("Tool returned invalid or non-text content")

                tool_output_content = observation.content[0].text

                if isinstance(tool_output_content, dict):
                    # It's already a dictionary, just use it
                    data = tool_output_content
                elif isinstance(tool_output_content, str):
                    # It's a string, so we need to parse it
                    data = json.loads(tool_output_content)
                else:
                    raise TypeError(f"Unknown tool output type: {type(tool_output_content)}")

                # 5. OUTBOUND Translation: (Tool -> Agent)
                if "detection" in tool_name and data.get("status") == "success" and not use_absolute_coords:
                    logger.info("Converting OUTBOUND coords for detection...")
                    data = await _convert_to_relative_coords(state, data, image_url_to_use)
                    logger.info("Converted detection output for agent.")

                real_image_url = data.get("output_image")
                _traj_data = data  # trajectory: structured tool output
                _traj_output_image_url = real_image_url  # trajectory

                if real_image_url and data.get("status") == "success":
                    logger.info("Tool generated a new image. Adding to state.")

                    fake_url = generate_fake_url(state, tool_name)
                    state["url_map"][fake_url] = real_image_url

                    # Append the new HumanMessage right after the ToolMessage
                    human_msg_to_add = HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": f"The tool successfully generated a new image. "
                                f"You can now refer to it as url: {fake_url}. It can"
                                f"works as tool input using the url.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": real_image_url},
                            },
                        ],
                    )

                final_tool_output = _post_process_tool_output(tool_name, data)

            except Exception as e:
                logger.error("Tool %s failed: %s", tool_name, e)
                final_tool_output = json.dumps(
                    {"status": "error", "message": f"Tool call failed: {e}"}
                )

            # --- Info gain collection: logprobs AFTER tool ---
            _info_gain_data = None
            if collect_info_gain and _info_gain_logprobs_before is not None:
                try:
                    from cv_agent.skillguard.info_gain import (
                        get_answer_logprobs, compute_label, build_trajectory_prefix, messages_to_api_format,
                    )
                    api_msgs_after = messages_to_api_format(state["messages"])
                    api_msgs_after.append({"role": "assistant", "content": final_tool_output})

                    logprobs_after = get_answer_logprobs(
                        api_msgs_after, info_gain_endpoint, info_gain_model
                    )
                    gt = state.get("_ground_truth", "") or state.get("ground_truth_for_ig", "")
                    _info_gain_data = compute_label(
                        _info_gain_logprobs_before, logprobs_after, gt
                    )
                    # Add trajectory prefix for probe training
                    _info_gain_data["trajectory_prefix"] = build_trajectory_prefix(
                        question=state.get("question", ""),
                        trajectory_steps=state.get("trajectory_steps", []),
                        pending_thinking=state.get("pending_thinking", ""),
                        pending_tool_name=tool_name,
                        pending_tool_params=tool_call["args"],
                    )
                    _info_gain_data["task_id"] = state.get("prefix", "")
                    _info_gain_data["step_idx"] = state.get("current_turn", 0) - 1
                    _info_gain_data["tool_name"] = tool_name
                    _info_gain_data["ground_truth"] = gt
                    logger.info("Info gain: label=%s for %s step %d",
                                _info_gain_data["tool_useful"], tool_name,
                                _info_gain_data["step_idx"])
                except Exception as e:
                    logger.warning("Info gain: failed to collect logprobs_after: %s", e)

            # --- trajectory: record this step ---
            _step_record = {
                "step": state.get("current_turn", 0) - 1,  # planner already incremented it
                "thinking": state.get("pending_thinking", ""),
                "belief": state.get("pending_belief", {}),
                "tool_name": tool_name,
                "tool_params": tool_call["args"],  # original agent-facing args
                "input_image_url": _traj_input_image_url,
                "tool_output": _traj_data,
                "output_image_url": _traj_output_image_url,
                "gate_decision": gate_info,
            }
            # Store thinking logprobs if available
            if state.get("pending_thinking_logprobs"):
                _step_record["thinking_logprobs"] = state["pending_thinking_logprobs"]
                _step_record["thinking_top_logprobs"] = state.get("pending_thinking_top_logprobs", [])
            if _info_gain_data is not None:
                _step_record["info_gain"] = _info_gain_data
            state.setdefault("trajectory_steps", []).append(_step_record)

            tool_msgs.append(ToolMessage(content=final_tool_output, tool_call_id=tool_call["id"]))

            if human_msg_to_add is not None:
                tool_msgs.append(human_msg_to_add)

            # --- SkillGuard: inject conflict notice if enabled ---
            if enable_skillguard:
                skillguard_notice = detect_conflict(
                    thinking=state.get("pending_thinking", ""),
                    tool_name=tool_name,
                    tool_output=_traj_data,
                )
                if skillguard_notice:
                    tool_msgs.append(HumanMessage(content=skillguard_notice))
                    state.setdefault("skillguard_log", []).append({
                        "step": state.get("current_turn", 0) - 1,
                        "tool_name": tool_name,
                        "notice": skillguard_notice,
                    })
                    logger.info("SkillGuard: notice injected at step %d for %s",
                                state.get("current_turn", 0) - 1, tool_name)

        # 8. Update state
        state["messages"].extend(tool_msgs)
        state["tool_usage"] = tool_usage
        return state

    return tool_executor


#
@routing_function_registry.register("cv_agent_react_should_continue")
def get_cv_react_should_continue(
    end_dst: str = "end",
    continue_dst: str = "continue",
    use_hard_turn_limit: bool = True,
) -> RoutingFunction:
    def should_continue(state: dict) -> str:
        print("--- SHOULD CONTINUE ---")

        last_msg = state["messages"][-1]
        current_turn = state["current_turn"]
        max_turns = state["max_turns"]

        if use_hard_turn_limit and current_turn > (max_turns + 2):
            print(f"[DEBUG] HARD CEILING HIT ({current_turn}). Force stopping.")
            return end_dst
        elif not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            print("[DEBUG] no tool call, go to end")
            return end_dst
        else:
            print("[DEBUG] tool call detected, continue")
            return continue_dst

    return should_continue


def _post_process_tool_output(tool_name: str, data: dict) -> str:
    """
    Cleans the raw tool output JSON, removing sensitive info (like real URLs)
    and simplifying the structure before sending it to the agent.
    """
    if data.get("status") != "success":
        # If the tool itself reported an error, just forward that.
        return json.dumps(
            {
                "status": "error",
                "message": data.get("message", "Tool reported an unspecified error."),
            }
        )

    # --- Tool-Specific Success Responses ---

    if "detection" in tool_name:
        # For detection, the agent only needs the detection results.

        results = data.get("output_text", data.get("detections", []))
        return json.dumps({"status": "success", "results": results})

    elif "ocr" in tool_name:
        try:
            # NOTE: Handle dynamic key for minerU ocr
            # data["content"] is a dict like {"main_test_image": {...}}
            # We just need the first item's value, regardless of its key.
            content_dict = data.get("content", {})

            if not content_dict or not isinstance(content_dict, dict) or len(content_dict) == 0:
                raise ValueError("OCR 'content' dictionary is empty or not a dict")

            first_item_value = list(content_dict.values())[0]
            text = first_item_value.get("md_content", "")
            is_empty = False

            # 1. Check for None or empty/whitespace string
            if text is None:
                is_empty = True
            elif isinstance(text, str):
                if not text.strip():
                    is_empty = True
                # 2. Check for the specific pattern: ![](...)
                # This regex looks for: starts with '![', has some alt text (optional),
                # ends with ']', then '(', then some url, then ')'
                elif re.match(r"^\!\[.*?\]\(.*?\)$", text.strip()):
                    is_empty = True

            if is_empty:
                return json.dumps(
                    {"status": "success", "extracted_text": "No text detected in this image."}
                )
            return json.dumps({"status": "success", "extracted_text": text})

        except (KeyError, IndexError, AttributeError, ValueError) as e:
            logger.error(f"Failed to parse 'minerU_ocr' structure: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Failed to parse OCR output structure. "
                    f"Expected content dict, but got: {data.get('content')}",
                }
            )

    else:
        return json.dumps(
            {"status": "success", "message": f"Tool '{tool_name}' executed successfully."}
        )

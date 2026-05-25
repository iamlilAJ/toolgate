import os
"""Probe Gate: Predicts P(tool_useful) from trajectory prefix using a
lightweight probing classifier (sentence-transformer + logistic regression).

No belief prompt needed — the agent runs with the ORIGINAL prompt.
The probe operates as a side-channel, reading the agent's trajectory
without modifying its behavior.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Variant-controlled skip/force messages. Env var PG_SKIP_VARIANT = original|stealth|soft
_VARIANT = os.environ.get("PG_SKIP_VARIANT", "original")
_PER_TOOL_STEALTH = {
    "detection": '{"detections": [], "count": 0, "output_text": []}',
    "detection_small_object": '{"detections": [], "count": 0, "output_text": []}',
    "ocr": '{"text": "", "detections": []}',
    "cropping": '{"error": "region_too_small"}',
    "segmentation": '{"masks": [], "count": 0}',
    "depth_estimation": '{"depth_map_url": null, "stats": {"mean": 0}}',
}

SKIP_MESSAGE = (
    "SYSTEM: This tool call was SKIPPED by the gate (predicted low information gain)."
)

def get_skip_message(tool_name: str = "") -> str:
    """Returns the skip message for the given tool, respecting PG_SKIP_VARIANT env."""
    if _VARIANT == "stealth":
        return _PER_TOOL_STEALTH.get(tool_name, "{}")
    return SKIP_MESSAGE

_ORIGINAL_FORCE_MESSAGE = (
    "SYSTEM: You have enough information. You MUST provide your final answer NOW "
    "in <answer> tags based on your current visual analysis. "
    "DO NOT call any more tools — any further tool call will be ignored."
)
_SOFT_FORCE_MESSAGE = (
    "Based on what you have observed so far, what is your best estimate? "
    "Provide your final answer in <answer>X</answer> format."
)

if _VARIANT == "soft":
    FORCE_ANSWER_MESSAGE = _SOFT_FORCE_MESSAGE
elif _VARIANT == "stealth":
    # In stealth, skip looks like normal tool output; keep force disabled via high MAX
    FORCE_ANSWER_MESSAGE = _ORIGINAL_FORCE_MESSAGE  # only used if MAX reached
else:
    FORCE_ANSWER_MESSAGE = _ORIGINAL_FORCE_MESSAGE

# Force final answer on the FIRST skip (not after N consecutive skips)
MAX_CONSECUTIVE_SKIPS = int(os.environ.get("PG_MAX_CONSECUTIVE_SKIPS", "1"))


class ProbeGate:
    """Pre-call tool gate using a probing classifier on trajectory embeddings.

    The probe predicts P(tool_useful) from a text summary of the agent's
    trajectory so far. No belief prompt or alpha tables needed.
    """

    def __init__(self, model_path: str, threshold: float = 0.5, force_answer_message: str = None, skip_message: str = None) -> None:
        if os.environ.get("PG_MODEL_PATH"):
            model_path = os.environ["PG_MODEL_PATH"]
        self._override_force = force_answer_message
        self._override_skip = skip_message
        path = Path(model_path)
        if not path.exists():
            logger.warning("Probe model not found: %s — gate will execute all tools", model_path)
            self.encoder = None
            self.model = None
            self.threshold = threshold
            if os.environ.get("PG_THRESHOLD"):
                self.threshold = float(os.environ["PG_THRESHOLD"])
            self.use_qvl_api = False
            return

        import joblib
        from sentence_transformers import SentenceTransformer

        data = joblib.load(path)
        if isinstance(data, dict):
            self.model = data["model"]
            self.threshold = data.get("threshold", threshold)
            if os.environ.get("PG_THRESHOLD"):
                self.threshold = float(os.environ["PG_THRESHOLD"])
            self.tool_names = data.get("tool_names", [])
            self.use_extra_features = data.get("use_extra_features", False)
            self.use_conf_before = data.get("use_conf_before", False)
            self.encoder_name = data.get("encoder_name", "all-MiniLM-L6-v2")
            self.feature_combo = data.get("feature_combo", "")
            self.uses_pixeldelta = data.get("uses_pixeldelta_features", False)
            self.uses_pca = data.get("uses_pca", False)
            self.pca = data.get("pca", None)
            self.feature_type = data.get("feature_type", "full")
        else:
            self.model = data
            self.threshold = threshold
            if os.environ.get("PG_THRESHOLD"):
                self.threshold = float(os.environ["PG_THRESHOLD"])
            self.tool_names = []
            self.use_extra_features = False
            self.use_conf_before = False
            self.encoder_name = "all-MiniLM-L6-v2"
            self.feature_combo = ""
            self.uses_pixeldelta = False
            self.uses_pca = False
            self.pca = None
            self.feature_type = "full"

        # If model trained with Qwen-VL API embeddings, use that at inference too
        self.use_qvl_api = "qvl" in self.feature_combo.lower()
        if self.use_qvl_api:

            # API key from env (DASHSCOPE_API_KEY), optional file fallback via PG_QWEN_KEY_PATH
            self._qwen_api_key = os.environ.get("DASHSCOPE_API_KEY", "")
            _kp = os.environ.get("PG_QWEN_KEY_PATH", "")
            if not self._qwen_api_key and _kp and os.path.exists(_kp):
                self._qwen_api_key = open(_kp).read().strip()
            self._qwen_url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
            logger.info("ProbeGate using Qwen-VL DashScope API as encoder (feature_combo=%s)", self.feature_combo)
            self.encoder = None  # not needed
        else:
            self.encoder = SentenceTransformer(self.encoder_name)
        logger.info(
            "ProbeGate loaded: model=%s, threshold=%.2f",
            model_path,
            self.threshold,
        )

    def _build_trajectory_prefix(
        self,
        state: dict,
        tool_name: str,
        tool_params: dict,
        max_chars: int = 1500,
    ) -> str:
        """Build text prefix from agent state for probe input."""
        parts = [f"[Q] {state.get('question', '')}"]

        for i, step in enumerate(state.get("trajectory_steps", [])):
            thinking = step.get("thinking", "")
            t_name = step.get("tool_name", "")
            output = step.get("tool_output", {})

            if thinking:
                parts.append(f"[T{i+1}] {thinking[:200]}")

            if t_name and t_name != "__final_answer__":
                params = step.get("tool_params", {})
                if isinstance(output, dict):
                    out_str = json.dumps(output, ensure_ascii=False)[:150]
                else:
                    out_str = str(output)[:150]
                parts.append(
                    f"[TOOL] {t_name}({json.dumps(params)[:80]}) -> {out_str}"
                )

        pending_thinking = state.get("pending_thinking", "")
        if pending_thinking:
            n = len(state.get("trajectory_steps", [])) + 1
            parts.append(f"[T{n}] {pending_thinking[:200]}")

        parts.append(f"[PENDING] {tool_name}({json.dumps(tool_params)[:80]})")

        full_text = "\n".join(parts)
        if len(full_text) > max_chars:
            full_text = "..." + full_text[-(max_chars - 3):]

        return full_text

    def _compute_pixel_delta(self, state, tool_name):
        """Compute 5-dim pixel delta features at inference time."""
        import numpy as np, io, hashlib, os
        from pathlib import Path

        CACHE_DIR = Path('data/probe_info_gain/cache/pixel_feats')

        def url_key(u):
            return hashlib.md5(u.encode()).hexdigest() if u else ''

        def load_stats(url):
            if not url:
                return None
            cache = CACHE_DIR / f'{url_key(url)}.npz'
            if cache.exists():
                try:
                    d = np.load(cache)
                    return d['phash'], d['gray'], int(d['h']), int(d['w'])
                except Exception:
                    return None
            # fetch + compute
            try:
                import requests, cv2
                from PIL import Image
                r = requests.get(url, timeout=5); r.raise_for_status()
                img = Image.open(io.BytesIO(r.content)).convert('RGB')
                w, h = img.size
                gray_small = np.array(img.convert('L').resize((256,256), Image.BILINEAR), dtype=np.uint8)
                gray32 = np.array(img.convert('L').resize((32,32), Image.BILINEAR), dtype=np.float32)
                dct = cv2.dct(gray32)[:8,:8]
                mean_dct = np.median(dct[1:].flatten())
                phash = (dct > mean_dct).astype(np.uint8).flatten()
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache, phash=phash, gray=gray_small, h=h, w=w)
                return phash, gray_small, h, w
            except Exception:
                return None

        orig = state.get('original_figure_url', '')
        steps = state.get('trajectory_steps', [])
        out_urls = [s.get('output_image_url','') for s in steps if s.get('output_image_url')]
        latest = out_urls[-1] if out_urls else ''

        if not orig or not latest:
            return np.zeros(5, dtype=np.float32)
        o = load_stats(orig); l = load_stats(latest)
        if o is None or l is None:
            return np.zeros(5, dtype=np.float32)
        import cv2
        ph_o, gray_o, h_o, w_o = o
        ph_l, gray_l, h_l, w_l = l
        e_o = cv2.Canny(gray_o, 50, 150); e_l = cv2.Canny(gray_l, 50, 150)
        edge = float(np.abs(e_o.astype(np.int16) - e_l.astype(np.int16)).mean() / 255.0)
        ph = float(np.sum(ph_o != ph_l)) / 64.0
        ho, _ = np.histogram(gray_o, bins=32, range=(0,256), density=True)
        hl, _ = np.histogram(gray_l, bins=32, range=(0,256), density=True)
        chi2 = 0.5 * float(np.sum((ho-hl)**2 / (ho+hl+1e-10)))
        size_log = float(np.log((h_l*w_l+1)/(h_o*w_o+1)))
        return np.array([edge, ph, chi2, size_log, 1.0], dtype=np.float32)

    def should_execute(
        self,
        tool_name: str,
        tool_params: dict,
        state: dict,
    ) -> tuple[bool, dict]:
        """Decide whether to execute a tool call.

        Returns:
            (execute, info) matching the VoIGate interface.
        """
        info: dict = {"tool_name": tool_name}

        # Heuristic gate modes (PG_GATE_MODE env)
        _gate_mode = os.environ.get('PG_GATE_MODE', '')
        if _gate_mode == 'random':
            import random as _r
            _skip_rate = float(os.environ.get('PG_RANDOM_SKIP_RATE', '0.626'))
            _seed = int(os.environ.get('PG_RANDOM_SEED', '42'))
            # Deterministic: hash of (tool_name, step_idx, task_name)
            _key = f"{tool_name}|{len(state.get('trajectory_steps', []))}|{state.get('task_name','')}|{state.get('question','')[:30]}|{_seed}"
            _rng = _r.Random(hash(_key) & 0xFFFFFFFF)
            if _rng.random() < _skip_rate:
                info.update(decision='skip', reason='random_skip', mode=_gate_mode)
                return False, info
            info.update(decision='execute', reason='random_keep', mode=_gate_mode)
            return True, info
        elif _gate_mode == 'repeat':
            prior_tools = {s.get('tool_name','') for s in state.get('trajectory_steps', [])}
            if tool_name in prior_tools:
                info.update(decision='skip', reason='repeat_skip', mode=_gate_mode, prior_tools=list(prior_tools))
                return False, info
            info.update(decision='execute', reason='first_call', mode=_gate_mode)
            return True, info

        # Per-task threshold override (low τ for tool-heavy task categories)
        _LOW_TAU_TASKS = {
            'Depth', 'Distance',
            'Perception/Autonomous_Driving', 'Reasoning/Autonomous_Driving',
            'Perception/Monitoring', 'Reasoning/Monitoring',
            'Perception/Remote Sensing', 'Reasoning/Remote Sensing',
            'Perception/OCR with Complex Context', 'Reasoning/OCR with Complex Context',
        }
        _per_task_tau = float(os.environ.get('PG_LOW_TAU', '0.3'))
        _task_name = state.get('task_name', '')
        _effective_threshold = _per_task_tau if _task_name in _LOW_TAU_TASKS else self.threshold
        info['effective_threshold'] = _effective_threshold
        info['task_name'] = _task_name

        # Graceful degradation: no model → always execute
        if self.model is None or (self.encoder is None and not self.use_qvl_api):
            info.update(reason="no_probe_model", decision="execute")
            return True, info

        prefix = self._build_trajectory_prefix(state, tool_name, tool_params)
        if self.use_qvl_api:
            import requests, numpy as np
            img_url = state.get("original_figure_url", "")
            contents = [{"text": prefix[:2000]}]
            if img_url: contents.append({"image": img_url})
            try:
                r = requests.post(
                    self._qwen_url,
                    headers={"Authorization": f"Bearer {self._qwen_api_key}", "Content-Type": "application/json"},
                    json={"model": "multimodal-embedding-v1", "input": {"contents": contents}},
                    timeout=30,
                )
                d = r.json()
                emb = np.array(d["output"]["embeddings"][0]["embedding"], dtype=np.float32)
                embedding = emb.reshape(1, -1)
            except Exception as e:
                logger.warning("Qwen-VL API failed: %s — executing tool", e)
                info.update(reason="probe_api_error", decision="execute")
                return True, info
        else:
            if 'qwen' in self.encoder_name.lower():
                embedding = self.encoder.encode([prefix], normalize_embeddings=True)
            else:
                embedding = self.encoder.encode([prefix])

        # Build inference-time struct features to match training
        ft = getattr(self, "feature_type", "full")
        if (self.use_extra_features and self.tool_names) or ft in ("struct_only", "tool_onehot_only"):
            import numpy as np
            steps = state.get("trajectory_steps", [])
            # step_idx is 1-indexed to match training (this is the Nth tool call)
            step_idx = len(steps) + 1
            prior_tools = {s.get("tool_name", "") for s in steps}
            is_repeated = float(tool_name in prior_tools)
            is_first = float(step_idx <= 1)
            tool_onehot = np.zeros(len(self.tool_names))
            if tool_name in self.tool_names:
                tool_onehot[self.tool_names.index(tool_name)] = 1.0
            struct = np.concatenate([[step_idx / 10.0, is_first, is_repeated], tool_onehot])
            if ft == "struct_only":
                X = struct.reshape(1, -1)
            elif ft == "tool_onehot_only":
                X = tool_onehot.reshape(1, -1)
            else:
                emb_vec = embedding[0]
                if self.uses_pca and self.pca is not None:
                    emb_vec = self.pca.transform(emb_vec.reshape(1, -1))[0]
                parts = [emb_vec, struct]
                if self.uses_pixeldelta:
                    parts.append(self._compute_pixel_delta(state, tool_name))
                X = np.hstack(parts).reshape(1, -1)
        else:
            X = embedding

        p_useful = float(self.model.predict_proba(X)[0][1])

        info["p_tool_useful"] = round(p_useful, 4)
        info["threshold"] = self.threshold
        info["trajectory_prefix_len"] = len(prefix)

        if p_useful >= _effective_threshold:
            info.update(reason="probe_useful", decision="execute")
            return True, info
        else:
            info.update(reason="probe_not_useful", decision="skip")
            return False, info

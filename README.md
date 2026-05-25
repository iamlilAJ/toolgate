# ToolGate: Reducing Perceptual Tool Calls in Vision-Language Agents with
Pre-Call Gating

Code for *"ToolGate: Reducing Perceptual Tool Calls in Vision-Language Agents with
Pre-Call Gating."*

A vision-language agent calls external tools (detection, OCR, cropping, depth,
segmentation) to answer hard visual questions. But many tool calls are useless
or even **harmful** (they flip a right answer to wrong). **ToolGate** is a
lightweight probe that, before each proposed tool call, predicts whether the
call will actually help — and skips it if not. The base VLM stays **frozen**;
only the small probe is trained.

## Key results

On a frozen Qwen3-VL agent across five benchmarks (V*, CVBench, HR-Bench-4k/8k,
MME), ToolGate **cuts tool calls from ~2.7 to ~1.0 per episode (~30% fewer
tokens) with no loss in accuracy** (30B: 70.8 to 72.5; +1.65pp). The probe
transfers across domains and backbones; see the paper for details.

## Repository layout

```
src/cv_agent/                 # the agent
  agents/                     #   ReAct / aggregator nodes
  skillguard/probe_gate.py    #   the ToolGate probe gate (PG_* env config)
  skillguard/info_gain.py     #   forced-answer logprob utility probing
  benchmark_loaders/          #   V*, CVBench, HR-Bench, MME loaders
  tools/                      #   MCP tool clients (detection/ocr/...)
  main_benchmark.py           #   evaluation driver
scripts/probing/              # data build, info-gain collection, probe training
configs/                      # model + tool endpoint configs (OmegaConf)
```

## Setup

```bash
uv sync                       # or: pip install -e .
cp .env.example .env          # then edit: point endpoints at YOUR servers
```

You provide your own OpenAI-compatible **VLM endpoint** (e.g. vLLM serving
Qwen3-VL) and **MCP tool servers**. Set the URLs in `.env` / the `configs/*.yaml`
(`base_url`, `server_url`). No API keys are committed; the optional
Qwen-Embedding probe encoder reads `DASHSCOPE_API_KEY` from the environment.

## Usage

```
# 1. Collect tool-call utility labels (forced-answer logprob probing)
python scripts/probing/collect_info_gain.py --dataset <bench> --output labels.jsonl

# 2. Train the probe (MiniLM + struct features, must_execute label)
python scripts/probing/train_skipsafe_ablation.py --input labels.jsonl --output probe.joblib

# 3. Evaluate with gating (probe path + threshold via env)
PG_MODEL_PATH=probe.joblib PG_THRESHOLD=0.5 \
  python -m cv_agent.main_benchmark --dataset <bench> --model <cfg> --mode agent
```

Gate behaviour is controlled by env vars read in `skillguard/probe_gate.py`
(`PG_MODEL_PATH`, `PG_THRESHOLD`, `PG_GATE_MODE`, ...).

> Note: some scripts under `scripts/probing/` assume a local data/results
> directory layout; adjust the paths near the top of each script for your setup.



## License

MIT — see [LICENSE](LICENSE).

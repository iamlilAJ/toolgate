"""SkillGuard: probe-based tool-call gating for frozen VLM agents."""

try:
    from cv_agent.skillguard.probe_gate import ProbeGate
except ImportError:  # sentence-transformers / sklearn not installed
    ProbeGate = None

__all__ = ["ProbeGate"]

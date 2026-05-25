from .openai import get_openai_model

try:
    from .ollama import get_ollama_model

    __all__ = ["get_ollama_model", "get_openai_model"]

except ImportError:
    __all__ = ["get_openai_model"]

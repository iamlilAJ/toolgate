try:
    from .langfuse import get_langfuse_callback_handler

    __all__ = ["get_langfuse_callback_handler"]
except ImportError:
    __all__ = []

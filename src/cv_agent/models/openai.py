"""LLMs hosted by OpenAI-compatible servers."""

from langchain_openai import ChatOpenAI

from cv_agent.core.registries import chat_model_registry


@chat_model_registry.register("openai_model")
def get_openai_model(
    model: str,
    base_url: str | None = None,
    **kwargs,
) -> ChatOpenAI:
    return ChatOpenAI(model=model, base_url=base_url, **kwargs)

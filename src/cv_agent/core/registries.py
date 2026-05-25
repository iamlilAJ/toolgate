from collections.abc import Awaitable, Callable

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langgraph.types import Command

#################################
# Base class for registries ###
#################################


class Registry[P]:  # P stands for Product
    def __init__(self) -> None:
        self._registry: dict[str, Callable[..., P]] = {}

    def register(self, name: str) -> Callable[[Callable[..., P]], Callable[..., P]]:
        def decorator(func: Callable[..., P]) -> Callable[..., P]:
            if name in self._registry:
                raise ValueError(f"{name} already registered.")

            self._registry[name] = func
            return func

        return decorator

    def get(self, name: str, **kwargs) -> P:
        if name not in self._registry:
            raise ValueError(f"{name} not found in {self.__class__.__name__}.")

        factory_fn = self._registry[name]
        return factory_fn(**kwargs)


#####################
# Tool Registry ###
#####################


# Inherit `Registry[P]' to make `self.__class__.__name__' unique for each registry
class ToolRegistry(Registry[StructuredTool]): ...


tool_registry = ToolRegistry()


###########################
# Chat Model Registry ###
###########################


class ChatModelRegistry(Registry[BaseChatModel]): ...


chat_model_registry = ChatModelRegistry()


#####################
# Node Registry ###
#####################

# A node is a function that takes the state and returns a dictionary of updates.
Node = Callable[[dict], Awaitable[dict | Command]]


class NodeRegistry(Registry[Node]): ...


node_registry = NodeRegistry()


#################################
# Routing Function Registry ###
#################################

# A routing function is the argument `path' to be passed to `add_conditional_edges(source, path,
# path_map)'.
RoutingFunction = Callable[[dict], str]


class RoutingFunctionRegistry(Registry[RoutingFunction]): ...


routing_function_registry = RoutingFunctionRegistry()


#################################
# Prompt Generator Registry ###
#################################

# A prompt generator takes the graph state as input and outputs a prompt.  For now we only consider
# text prompts, and leave the processing of other types to the node itself.
PromptGenerator = Callable[[dict], str]


class PromptGeneratorRegistry(Registry[PromptGenerator]): ...


prompt_generator_registry = PromptGeneratorRegistry()


#################################
# Callback Handler Registry ###
#################################


class CallbackHandlerRegistry(Registry[BaseCallbackHandler]): ...


callback_handler_registry = CallbackHandlerRegistry()

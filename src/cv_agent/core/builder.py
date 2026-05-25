from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from omegaconf import DictConfig, OmegaConf

from cv_agent.core.registries import (
    chat_model_registry,
    node_registry,
    routing_function_registry,
    tool_registry,
)
from cv_agent.core.state import create_agent_state


class GraphBuilder:
    """
    Builds a LangGraph workflow from a YAML configuration.

    This class is the bridge between a declarative YAML file and an executable
    LangGraph graph. It handles state creation, node registration, and edge
    construction.
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.state_definition = create_agent_state(config.state)
        self.graph = StateGraph(self.state_definition)

    def build(self) -> CompiledStateGraph:
        """Constructs and compiles the graph based on the configuration."""
        self._init_tools()
        self._init_chat_models()
        self._add_nodes()
        self._set_entry_point()
        self._add_edges()
        return self.graph.compile()

    def _init_chat_models(self) -> None:
        if "llms" in self.config:  # discard "llm" if "llms" exists
            self._llms = {}
            for llm_config in self.config.llms:
                self._llms[llm_config.id] = chat_model_registry.get(
                    llm_config.name, **llm_config.parameters
                )
            # default to the first one for each node
            self._chat_model = self._llms[self.config.llms[0].id]
        else:
            self._chat_model = chat_model_registry.get(
                self.config.llm.name, **self.config.llm.parameters
            )
            self._llms = {"default": self._chat_model}

    def _init_tools(self) -> None:
        self._tools_by_registered_name = {
            tool.name: tool_registry.get(tool.name, **tool.parameters) for tool in self.config.tools
        }
        self._tools = list(self._tools_by_registered_name.values())
        self._tools_by_name = {tool.name: tool for tool in self._tools}

    def _add_nodes(self):
        """Adds nodes to the graph from the config."""
        for name, node_config in self.config.workflow.nodes.items():
            node_parameters = node_config.get("parameters", {})

            # Chat model for this node.  Defaults to the first one of "config.llms" if not specified
            # in config.
            if "llm" in node_parameters:
                llm_id = node_parameters.pop("llm")
                if llm_id not in self._llms:
                    raise ValueError(f"Unknown chat model: {llm_id}. Register it in config first.")
                chat_model = self._llms[llm_id]
            else:
                chat_model = self._chat_model

            # These are tools provided for agent nodes.
            if "tools" in node_parameters:
                tool_config = node_parameters.pop("tools")
                tools = [self._tools_by_registered_name[name] for name in tool_config]
            else:
                tools = []

            # We must divide provided tools and executable ones because tool executor nodes must
            # have access to all tools by default.  In this way, users don't need to explicitly add
            # all tools to tool nodes in config.  If they don't want tool nodes to execute some,
            # they should remove them from the `tools' config in the first place (it makes no sense
            # to include them in the config but you don't want them to be executed).
            node_func = node_registry.get(
                node_config.name,
                chat_model=chat_model,  # required by agent nodes
                provided_tools=tools,  # required by agent nodes
                executable_tools=self._tools_by_name,  # required by tool executor nodes
                **node_parameters,
            )
            self.graph.add_node(name, node_func)  # type: ignore

    def _set_entry_point(self):
        """Sets the graph's entry point from the config."""
        entry_point = self.config.workflow.entry_point
        self.graph.add_edge(START, entry_point)

    def _add_edges(self):
        """Adds edges to the graph from the config."""
        for edge_config in self.config.workflow.edges:
            source = edge_config.source

            if "condition" in edge_config:
                # This is a conditional edge
                #  Extract parameters for the routing function
                condition_config = edge_config.condition
                function_name = condition_config.function
                function_params = condition_config.get("parameters", {})
                # Pass these parameters when getting the function from the registry
                condition_func = routing_function_registry.get(function_name, **function_params)

                mapping = OmegaConf.to_container(edge_config.condition.mapping)
                assert isinstance(mapping, dict)

                # Ensure END is correctly referenced if it's a string in YAML
                for key, target in mapping.items():
                    if target == "END":
                        mapping[key] = END

                self.graph.add_conditional_edges(source, condition_func, mapping)  # type: ignore

            else:
                # This is a standard edge
                target = edge_config.target
                if target == "END":
                    self.graph.add_edge(source, END)
                else:
                    self.graph.add_edge(source, target)

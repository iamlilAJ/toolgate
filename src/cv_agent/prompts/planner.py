from pathlib import Path

from jinja2 import Template

from cv_agent.core.registries import PromptGenerator, prompt_generator_registry

DEFAULT_TEMPLATE_PATH = Path(__file__).parent / "templates" / "planner_react.j2"


@prompt_generator_registry.register("planner")
def get_planner_prompt_generator(template_path: Path | str = DEFAULT_TEMPLATE_PATH) -> PromptGenerator:
    template_path = Path(template_path)
    template = Template(template_path.read_text())

    def generator(state: dict) -> str:
        if "question" in state:
            question = state["question"]
        else:
            raise ValueError(f"Cannot find question in state '{repr(state)}'.")

        if "current_turn" in state:
            current_turn = state["current_turn"]
        else:
            raise ValueError(f"Cannot find current_turn in state '{repr(state)}'.")

        if "max_turns" in state:
            max_turns = state["max_turns"]
        else:
            raise ValueError(f"Cannot find max_turns in state '{repr(state)}'.")

        if "original_figure_url" in state:
            original_figure_url = state["original_figure_url"]
        else:
            raise ValueError(f"Cannot find original_figure_url in state '{repr(state)}'.")

        return template.render(
            question=question,
            max_turns=max_turns,
            current_turn=current_turn,
            original_figure_url=original_figure_url,
        ).strip()

    return generator

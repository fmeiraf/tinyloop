from pathlib import Path

import yaml
from jinja2 import Environment, StrictUndefined


def render_base_prompts(yaml_path: str, **overrides) -> dict[str, str]:
    """
    Load a YAML prompt file with `system` and `user` templates,
    render them with Jinja2, using remaining top-level keys as defaults.
    """
    data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    try:
        sys_tmpl = data.pop("system")
        usr_tmpl = data.pop("user")
    except KeyError as e:
        raise KeyError(f"Missing required key: {e.args[0]!r} in {yaml_path}")

    # Everything else at top level becomes a default variable
    defaults = data

    # StrictUndefined → fail fast if you forget a variable
    env = Environment(
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,  # plain text prompts
    )

    ctx = {**defaults, **overrides}
    system = env.from_string(sys_tmpl).render(**ctx)
    user = env.from_string(usr_tmpl).render(**ctx)
    return {"system": system, "user": user, "context_used": ctx}


def render_specific_prompt(yaml_path: str, variable_name: str, **overrides) -> str:
    """
    Render a specific prompt from a YAML file.

    Args:
        yaml_path (str): The path to the YAML file.
        variable_name (str): The name of the variable to render.
        **overrides: Additional overrides to pass to the prompt.

    Returns:
        The rendered prompt.
    """
    data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    try:
        tmpl = data[variable_name]
    except KeyError as e:
        raise KeyError(f"Missing required key: {e.args[0]!r} in {yaml_path}")

    env = Environment(
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,  # plain text prompts
    )
    return env.from_string(tmpl).render(**overrides)


def render_specific_prompts(
    yaml_path: str, variable_names: list[str], **overrides
) -> dict[str, str]:
    """
    Render a list of specific prompts from a specific YAML file.

    Args:
        yaml_path (str): The path to the YAML file.
        variable_names (list[str]): The names of the variables to render.
        **overrides: Additional overrides to pass to the prompts.

    Returns:
        A dictionary of the rendered prompts.
    """
    data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    try:
        tmpls = {variable_name: data[variable_name] for variable_name in variable_names}
    except KeyError as e:
        raise KeyError(f"Missing required key: {e.args[0]!r} in {yaml_path}")

    env = Environment(
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,  # plain text prompts
    )
    return {
        variable_name: env.from_string(tmpl).render(**overrides)
        for variable_name, tmpl in tmpls.items()
    }


class PromptRenderer:
    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
        self.env = Environment(
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,  # plain text prompts
        )

    def render(self, variable_name: str, **overrides) -> str:
        try:
            tmpl = self.data[variable_name]
        except KeyError:
            raise KeyError(f"Missing required key: {variable_name} in {self.yaml_path}")
        return self.env.from_string(tmpl).render(**overrides)

    def render_many(self, variable_names: list[str], **overrides) -> dict[str, str]:
        try:
            tmpls = {
                variable_name: self.data[variable_name]
                for variable_name in variable_names
            }
        except KeyError as e:
            raise KeyError(f"Missing required key: {e.args[0]!r} in {self.yaml_path}")
        return {
            variable_name: self.env.from_string(tmpl).render(**overrides)
            for variable_name, tmpl in tmpls.items()
        }

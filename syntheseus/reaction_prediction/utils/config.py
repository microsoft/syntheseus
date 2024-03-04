import argparse
import sys
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

from omegaconf import DictConfig, ListConfig, OmegaConf

R = TypeVar("R")


def get_config(
    argv: Optional[List[str]],
    config_cls: Callable[..., R],
    defaults: Optional[Dict[str, Any]] = None,
) -> R:
    """
    Utility function to get `OmegaConf` config options.

    Args:
        argv: Either a list of command line arguments to parse, or `None`. If `None`, this argument
            is set from `sys.argv`.
        config_cls: Dataclass object specifying config structure (i.e. which fields to expect in the
            config). It should be the class itself, not an instance of the class.

    Returns:
        Config object, which will pass as an instance of `config_cls` among other things. Note: the
        type for this could be specified more carefully, but `OmegaConf`'s typing system is a bit
        complex. Search `OmegaConf`'s docs for "structured" for more info.
    """

    if argv is None:
        argv = sys.argv[1:]
    # Parse command line arguments
    parser = argparse.ArgumentParser(allow_abbrev=False)  # prevent prefix matching issues
    parser.add_argument(
        "--config",
        type=str,
        action="append",
        default=list(),
        help="Path to a yaml config file. "
        "Argument can be repeated multiple times, with later configs overwriting previous ones.",
    )
    args, config_changes = parser.parse_known_args(argv)

    # Read configs from defaults, file and command line
    conf_yamls: List[Union[DictConfig, ListConfig]] = []
    if defaults:
        conf_yamls = [OmegaConf.create(defaults)]

    conf_yamls += [OmegaConf.load(c) for c in args.config]
    conf_cli = OmegaConf.from_cli(config_changes)

    # Make merged config options
    # CLI options take priority over YAML file options
    schema = OmegaConf.structured(config_cls)
    config = OmegaConf.merge(schema, *conf_yamls, conf_cli)
    OmegaConf.set_readonly(config, True)  # should not be written to
    return cast(R, config)


def get_error_message_for_missing_value(name: str, possible_values: List[str]) -> str:
    return f"{name} should be set to one of [{', '.join(possible_values)}]"

"""Script for running end-to-end retrosynthetic search.

The supported single-step model types are listed in `syntheseus/cli/eval_single_step.py`;
each can be combined with Retro*, MCTS, or PDVN to perform search.

Example invocation:
    python ./syntheseus/cli/search.py \
        inventory_smiles_file=[INVENTORY_SMILES_FILE_PATH] \
        search_target="NC1=Nc2ccc(F)cc2C2CCCC12" \
        model_class=LocalRetro \
        model_dir=[MODEL_DIR] \
        time_limit_s=60
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import pickle
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Iterator, List, Optional, cast

import yaml
from omegaconf import MISSING, DictConfig, OmegaConf
from tqdm import tqdm

from syntheseus import Molecule
from syntheseus.reaction_prediction.inference.config import BackwardModelConfig
from syntheseus.reaction_prediction.utils.config import get_config as cli_get_config
from syntheseus.reaction_prediction.utils.misc import set_random_seed
from syntheseus.reaction_prediction.utils.model_loading import get_model
from syntheseus.search import INT_INF
from syntheseus.search.algorithms.best_first.retro_star import RetroStarSearch
from syntheseus.search.algorithms.mcts import base as mcts_base
from syntheseus.search.algorithms.mcts.molset import MolSetMCTS
from syntheseus.search.algorithms.pdvn import PDVN_MCTS
from syntheseus.search.analysis.route_extraction import iter_routes_time_order
from syntheseus.search.analysis.solution_time import get_first_solution_time
from syntheseus.search.graph.and_or import AndOrGraph
from syntheseus.search.graph.molset import MolSetGraph
from syntheseus.search.mol_inventory import SmilesListInventory
from syntheseus.search.node_evaluation import common as node_evaluation_common
from syntheseus.search.utils.misc import lookup_by_name

try:
    # Try to import the visualization code, which will work only if `graphviz` is installed.
    from syntheseus.search.visualization import visualize_andor, visualize_molset

    VISUALIZATION_CODE_IMPORTED = True
except ModuleNotFoundError:
    VISUALIZATION_CODE_IMPORTED = False

logger = logging.getLogger(__file__)


@dataclass
class RetroStarConfig:
    max_expansion_depth: int = 10

    value_function_class: str = "ConstantNodeEvaluator"
    value_function_kwargs: Dict[str, Any] = field(default_factory=lambda: {"constant": 0.0})

    and_node_cost_fn_class: str = "ReactionModelLogProbCost"
    and_node_cost_fn_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCTSConfig:
    max_expansion_depth: int = 20

    value_function_class: str = "ConstantNodeEvaluator"
    value_function_kwargs: Dict[str, Any] = field(default_factory=lambda: {"constant": 0.5})

    reward_function_class: str = "HasSolutionValueFunction"
    reward_function_kwargs: Dict[str, Any] = field(default_factory=dict)

    policy_class: str = "ReactionModelProbPolicy"
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)

    bound_constant: float = 1.0
    bound_function_class: str = "pucb_bound"


@dataclass
class PDVNConfig:
    max_expansion_depth: int = 10

    value_function_syn_class: str = "ConstantNodeEvaluator"
    value_function_syn_kwargs: Dict[str, Any] = field(default_factory=lambda: {"constant": 0.5})

    value_function_cost_class: str = "ConstantNodeEvaluator"
    value_function_cost_kwargs: Dict[str, Any] = field(default_factory=lambda: {"constant": 0.0})

    and_node_cost_fn_class: str = "ConstantNodeEvaluator"
    and_node_cost_fn_kwargs: Dict[str, Any] = field(default_factory=lambda: {"constant": 0.1})

    policy_class: str = "ReactionModelProbPolicy"
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)

    c_dead: float = 5.0
    bound_constant: float = 1e2
    bound_function_class: str = "pucb_bound"


@dataclass
class BaseSearchConfig:
    # Molecule(s) to search for (either as a single explicit SMILES or a file)
    search_target: str = MISSING
    search_targets_file: str = MISSING

    inventory_smiles_file: str = MISSING  # Purchasable molecules
    results_dir: str = "."  # Directory to save the results in
    append_timestamp_to_dir: bool = True  # Whether to append the current time to directory name

    # By default limit search time (but set very high iteration limits just in case)
    time_limit_s: float = 600
    limit_reaction_model_calls: int = 1_000_000
    limit_iterations: int = 1_000_000
    limit_graph_nodes: int = INT_INF
    prevent_repeat_mol_in_trees: bool = True
    stop_on_first_solution: bool = False

    use_gpu: bool = True  # Whether to use a GPU
    canonicalize_inventory: bool = False  # Whether to canonicalize the inventory SMILES

    # Fields configuring the reaction model (on top of the arguments from `BackwardModelConfig`)
    num_top_results: int = 50  # Number of results to request
    reaction_model_use_cache: bool = True  # Whether to cache the results

    # Fields configuring the search algorithm
    search_algorithm: str = "retro_star"  # "retro_star", "mcts", or "pdvn"
    retro_star_config: RetroStarConfig = RetroStarConfig()
    mcts_config: MCTSConfig = MCTSConfig()
    pdvn_config: PDVNConfig = PDVNConfig()

    # Fields configuring what to save after the run
    save_graph: bool = True  # Whether to save the full reaction graph (can be large)
    num_routes_to_plot: int = 5  # Number of routes to extract and plot for a quick check


@dataclass
class SearchConfig(BackwardModelConfig, BaseSearchConfig):
    """Config for running search for given search targets."""

    pass


def run_from_config(config: SearchConfig) -> Path:
    set_random_seed(0)

    print("Running search with the following config:")
    print(config)

    if config.num_routes_to_plot > 0 and not VISUALIZATION_CODE_IMPORTED:
        raise ValueError(
            "Could not import visualization code (likely `viz` dependencies are not installed); "
            "please install missing dependencies or set `num_routes_to_plot=0`"
        )

    search_target, search_targets_file = [
        cast(DictConfig, config).get(key) for key in ["search_target", "search_targets_file"]
    ]

    if not ((search_target is None) ^ (search_targets_file is None)):
        raise ValueError(
            "Exactly one of 'search_target' and 'search_targets_file' should be provided"
        )

    # Prepare the search targets
    search_targets: List[str] = []
    if search_target is not None:
        search_targets = [search_target]
    else:
        with open(config.search_targets_file, "rt") as f_targets:
            search_targets = [line.strip() for line in f_targets]

    if not config.save_graph and config.num_routes_to_plot == 0:
        logger.warning(
            "Neither 'save_graph' nor 'num_routes_to_plot' is set; output saved will be minimal"
        )

    # Load the single-step model
    search_rxn_model = get_model(  # type: ignore
        config,
        batch_size=1,
        num_gpus=int(config.use_gpu),
        use_cache=config.reaction_model_use_cache,
        default_num_results=config.num_top_results,
    )

    # Set up the inventory
    with open(config.inventory_smiles_file, "rt") as f_inventory:
        inventory_smiles = [line.strip() for line in f_inventory]
    mol_inventory = SmilesListInventory(
        inventory_smiles, canonicalize=config.canonicalize_inventory
    )

    alg_kwargs: Dict[str, Any] = dict(reaction_model=search_rxn_model, mol_inventory=mol_inventory)
    alg_kwargs.update(
        **{
            key: cast(DictConfig, config).get(key)
            for key in [
                "time_limit_s",
                "limit_reaction_model_calls",
                "limit_iterations",
                "limit_graph_nodes",
                "prevent_repeat_mol_in_trees",
                "stop_on_first_solution",
            ]
        }
    )

    def build_node_evaluator(key: str) -> None:
        # Build a node evaluator based on chosen class and args
        alg_kwargs[key] = lookup_by_name(node_evaluation_common, alg_kwargs[f"{key}_class"])(
            **alg_kwargs[f"{key}_kwargs"]
        )

        # Delete the arguments to avoid passing them into the algorithm's constructor downstream
        del alg_kwargs[f"{key}_class"]
        del alg_kwargs[f"{key}_kwargs"]

    alg: Any = None
    if config.search_algorithm == "retro_star":
        alg_kwargs.update(cast(Dict[str, Any], OmegaConf.to_container(config.retro_star_config)))
        build_node_evaluator("value_function")
        build_node_evaluator("and_node_cost_fn")

        alg = RetroStarSearch(**alg_kwargs)
    elif config.search_algorithm == "mcts":
        alg_kwargs.update(cast(Dict[str, Any], OmegaConf.to_container(config.mcts_config)))
        build_node_evaluator("value_function")
        build_node_evaluator("reward_function")
        build_node_evaluator("policy")

        alg_kwargs["bound_function"] = lookup_by_name(mcts_base, alg_kwargs["bound_function_class"])
        del alg_kwargs["bound_function_class"]

        alg = MolSetMCTS(**alg_kwargs)
    elif config.search_algorithm == "pdvn":
        alg_kwargs.update(cast(Dict[str, Any], OmegaConf.to_container(config.pdvn_config)))
        build_node_evaluator("value_function_syn")
        build_node_evaluator("value_function_cost")
        build_node_evaluator("and_node_cost_fn")
        build_node_evaluator("policy")

        alg_kwargs["bound_function"] = lookup_by_name(mcts_base, alg_kwargs["bound_function_class"])
        del alg_kwargs["bound_function_class"]

        alg = PDVN_MCTS(**alg_kwargs)
    else:
        raise NotImplementedError(f"Unsupported search algorithm {config.search_algorithm}")

    # Prepare the output directory
    results_dir_top_level = Path(config.results_dir)

    dirname = config.model_class.name
    if config.append_timestamp_to_dir:
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        dirname += f"_{str(timestamp)}"

    results_dir_current_run = results_dir_top_level / dirname

    logger.info("Setup completed")
    num_targets = len(search_targets)

    all_stats: List[Dict[str, Any]] = []
    for idx, smiles in enumerate(tqdm(search_targets)):
        logger.info(f"Running search for target {smiles}")

        if num_targets == 1:
            results_dir = results_dir_current_run
        else:
            results_dir = results_dir_current_run / str(idx)

        results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Outputs will be saved under {results_dir}")

        results_lock_path = results_dir / ".lock"
        results_stats_path = results_dir / "stats.json"

        if results_lock_path.exists():
            paths = [path for path in results_dir.iterdir() if path.is_file()]
            logger.warning(
                f"Lockfile was found which means the last run failed, purging {len(paths)} files"
            )

            for path in paths:
                path.unlink()
        elif results_stats_path.exists():
            with open(results_stats_path, "rt") as f_stats:
                stats = json.load(f_stats)
                if stats.get("index") != idx or stats.get("smiles") != smiles:
                    raise RuntimeError(
                        f"Data present under {results_dir} does not match the current run"
                    )

                all_stats.append(stats)

            logger.info("Search results already exist, skipping")
            continue

        results_lock_path.touch()

        alg.reset()
        output_graph, _ = alg.run_from_mol(Molecule(smiles))
        logger.info(f"Finished search for target {smiles}")

        # Time of first solution (rxn model calls)
        for node in output_graph.nodes():
            node.data["analysis_time"] = node.data["num_calls_rxn_model"]
        soln_time_rxn_model_calls = get_first_solution_time(output_graph)

        # Time of first solution (wallclock)
        for node in output_graph.nodes():
            node.data["analysis_time"] = (
                node.creation_time - output_graph.root_node.creation_time
            ).total_seconds()
        soln_time_wallclock = get_first_solution_time(output_graph)

        stats = {
            "index": idx,
            "smiles": smiles,
            "rxn_model_calls_used": alg.reaction_model.num_calls(),
            "num_nodes_in_final_tree": len(output_graph),
            "soln_time_rxn_model_calls": soln_time_rxn_model_calls,
            "soln_time_wallclock": soln_time_wallclock,
        }

        all_stats.append(stats)
        logger.info(pformat(stats))

        with open(results_stats_path, "wt") as f_stats:
            f_stats.write(json.dumps(stats, indent=2))

        if config.save_graph:
            with open(results_dir / "graph.pkl", "wb") as f_graph:
                pickle.dump(output_graph, f_graph)

        if config.num_routes_to_plot > 0:
            # Extract some synthesis routes in the order they were found
            logger.info(f"Extracting up to {config.num_routes_to_plot} routes for analysis")

            # TODO(kmaziarz): Add options to extract a diverse (or otherwise interesting) subset.
            routes: Iterator = iter_routes_time_order(
                output_graph, max_routes=config.num_routes_to_plot
            )

            for route_idx, route in enumerate(routes):
                with open(results_dir / f"route_{route_idx}.pkl", "wb") as f_route:
                    pickle.dump(route, f_route)

                visualize_kwargs: Dict[str, Any] = dict(
                    graph=output_graph,
                    filename=str(results_dir / f"route_{route_idx}.pdf"),
                    nodes=route,
                )

                if isinstance(output_graph, AndOrGraph):
                    visualize_andor(**visualize_kwargs)
                elif isinstance(output_graph, MolSetGraph):
                    visualize_molset(**visualize_kwargs)
                else:
                    assert False

        results_lock_path.unlink()
        del results_dir

    if num_targets > 1:
        logger.info(f"Writing summary statistics across all {num_targets} targets")
        combined_stats: Dict[str, float] = dict(
            num_targets=num_targets,
            num_solved_targets=sum(stats["soln_time_wallclock"] != math.inf for stats in all_stats),
        )

        for key in [
            "rxn_model_calls_used",
            "num_nodes_in_final_tree",
            "soln_time_rxn_model_calls",
            "soln_time_wallclock",
        ]:
            values = [stats[key] for stats in all_stats]
            combined_stats[f"average_{key}"] = statistics.mean(values)
            combined_stats[f"median_{key}"] = statistics.median(values)

        logger.info(pformat(combined_stats))

        with open(results_dir_current_run / "stats.json", "wt") as f_combined_stats:
            f_combined_stats.write(json.dumps(combined_stats, indent=2))

    return results_dir_current_run


def main(argv: Optional[List[str]] = None, config_cls: Any = SearchConfig) -> Path:
    config = cli_get_config(argv=argv, config_cls=config_cls)

    def _warn_will_not_use_defaults(message: str) -> None:
        logger.warning(f"{message}; no model-specific search hyperparameters will be used")

    defaults_file_path = Path(__file__).parent / "search_config.yml"
    if not defaults_file_path.exists():
        _warn_will_not_use_defaults(f"File {defaults_file_path} does not exist")
    else:
        with open(defaults_file_path, "rt") as f_defaults:
            defaults = yaml.safe_load(f_defaults)

        if config.search_algorithm not in defaults:
            _warn_will_not_use_defaults(
                f"Hyperparameter defaults file has no entry for {config.search_algorithm}"
            )
        else:
            search_algorithm_defaults = defaults[config.search_algorithm]

            model_name = config.model_class.name
            if model_name not in search_algorithm_defaults:
                _warn_will_not_use_defaults(
                    f"Hyperparameter defaults file has no entry for {model_name}"
                )
            else:
                relevant_defaults = search_algorithm_defaults[model_name]
                logger.info(
                    f"Using hyperparameter defaults from {defaults_file_path}: {relevant_defaults}"
                )

                # We now parse the config again (we could not have included the defaults earlier as
                # we did not know the search algorithm and model class before the first parsing).
                config = cli_get_config(
                    argv=argv,
                    config_cls=config_cls,
                    defaults={f"{config.search_algorithm}_config": relevant_defaults},
                )

    return run_from_config(config)


if __name__ == "__main__":
    main()

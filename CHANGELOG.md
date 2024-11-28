# Changelog

All notable changes to the project are documented in this file.

The format follows [Common Changelog](https://common-changelog.org/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2024-11-28

### Changed

- Update GLN to Python 3.9 ([#103](https://github.com/microsoft/syntheseus/pull/103)) ([@kmaziarz])

### Added

- Reuse search results when given a partially filled directory ([#98](https://github.com/microsoft/syntheseus/pull/98)) ([@kmaziarz])
- Expose `stop_on_first_solution` as a CLI flag ([#100](https://github.com/microsoft/syntheseus/pull/100)) ([@kmaziarz])
- Add an option to set time limit for route extraction ([#104](https://github.com/microsoft/syntheseus/pull/104)) ([@kmaziarz])
- Extend single-step evaluation with stereo-agnostic results ([#102](https://github.com/microsoft/syntheseus/pull/102)) ([@kmaziarz])

### Fixed

- Install `torch_scatter` from conda instead of pip ([#110](https://github.com/microsoft/syntheseus/pull/110)) ([@kmaziarz])
- Remove dependency on `typing_extensions` for python >= 3.8 ([#107](https://github.com/microsoft/syntheseus/pull/107)) ([@austint])
- Pin further Chemformer dependencies to avoid `torch` reinstallation ([#108](https://github.com/microsoft/syntheseus/pull/108)) ([@kmaziarz])
- Shift the `pandas` dependency to the external model packages ([#94](https://github.com/microsoft/syntheseus/pull/94)) ([@kmaziarz])
- Fix constructor arguments in `ParallelReactionModel` ([#96](https://github.com/microsoft/syntheseus/pull/96)) ([@kmaziarz])

## [0.4.1] - 2024-05-04

### Fixed

- Fix incorrectly uploaded RetroKNN weights ([#91](https://github.com/microsoft/syntheseus/pull/91)) ([@kmaziarz])
- Fix GLN weights and issues in its model wrapper ([#92](https://github.com/microsoft/syntheseus/pull/92)) ([@kmaziarz])

## [0.4.0] - 2024-04-10

### Changed

- Merge reaction and reaction model base classes in `search` and `reaction_prediction` ([#63](https://github.com/microsoft/syntheseus/pull/63), [#67](https://github.com/microsoft/syntheseus/pull/67), [#73](https://github.com/microsoft/syntheseus/pull/73), [#74](https://github.com/microsoft/syntheseus/pull/74), [#76](https://github.com/microsoft/syntheseus/pull/76), [#84](https://github.com/microsoft/syntheseus/pull/84)) ([@austint], [@kmaziarz])
- Make reaction models return `Sequence[Reaction]` instead of `PredictionList` objects ([#61](https://github.com/microsoft/syntheseus/pull/61)) ([@austint])
- Suppress the remaining noisy logs and warnings coming from single-step models ([#53](https://github.com/microsoft/syntheseus/pull/53)) ([@kmaziarz])
- Improve efficiency and logging of retro* algorithm ([#62](https://github.com/microsoft/syntheseus/pull/62)) ([@austint])
- Improve error handling in single-step evaluation and allow CLI to use the default checkpoints ([#75](https://github.com/microsoft/syntheseus/pull/75)) ([@kmaziarz])
- Make basic classes from `interface` importable from top-level ([#81](https://github.com/microsoft/syntheseus/pull/81)) ([@austint])

### Added

- Integrate the Graph2Edits model ([#65](https://github.com/microsoft/syntheseus/pull/65), [#66](https://github.com/microsoft/syntheseus/pull/66)) ([@kmaziarz])
- Improve the docs and add tutorials ([#54](https://github.com/microsoft/syntheseus/pull/54), [#77](https://github.com/microsoft/syntheseus/pull/77), [#78](https://github.com/microsoft/syntheseus/pull/78), [#79](https://github.com/microsoft/syntheseus/pull/79), [#82](https://github.com/microsoft/syntheseus/pull/82)) ([@kmaziarz], [@austint])
- Add random search algorithm as a simple baseline ([#83](https://github.com/microsoft/syntheseus/pull/83)) ([@austint])
- Add optional argument `limit_graph_nodes` to base search algorithm class to stop search after the search graph exceeds a certain number of nodes ([#85](https://github.com/microsoft/syntheseus/pull/85)) ([@austint])

### Fixed

- Fix small issues in Chemformer, MEGAN and RootAligned ([#80](https://github.com/microsoft/syntheseus/pull/80)) ([@kmaziarz])
- Get all single-step models to work on CPU ([#57](https://github.com/microsoft/syntheseus/pull/57)) ([@kmaziarz])
- Make the data loader class work with relative paths ([#69](https://github.com/microsoft/syntheseus/pull/69)) ([@kmaziarz])

## [0.3.0] - 2023-12-19

### Changed

- Simplify single-step model setup ([#41](https://github.com/microsoft/syntheseus/pull/41), [#48](https://github.com/microsoft/syntheseus/pull/48)) ([@kmaziarz])
- Refactor single-step evaluation script and move it to cli/ ([#43](https://github.com/microsoft/syntheseus/pull/43)) ([@kmaziarz])
- Return model predictions as dataclasses instead of pydantic models ([#47](https://github.com/microsoft/syntheseus/pull/47)) ([@kmaziarz])
- Make the package compatible with PyPI ([#50](https://github.com/microsoft/syntheseus/pull/50)) ([@kmaziarz])

### Added

- Add a general CLI endpoint ([#44](https://github.com/microsoft/syntheseus/pull/44)) ([@kmaziarz])
- Add support for PDVN to the search CLI ([#46](https://github.com/microsoft/syntheseus/pull/46)) ([@fiberleif])
- Add initial static documentation ([#45](https://github.com/microsoft/syntheseus/pull/45)) ([@kmaziarz])

## [0.2.0] - 2023-11-21

### Changed

- Select search hyperparameters depending on which algorithm and single-step model are used ([#30](https://github.com/microsoft/syntheseus/pull/30)) ([@kmaziarz])
- Improve the heuristic used for estimating diversity ([#22](https://github.com/microsoft/syntheseus/pull/22), [#28](https://github.com/microsoft/syntheseus/pull/28)) ([@kmaziarz])

### Added

- Add code for PDVN MCTS and extracting training data for policies and value functions ([#8](https://github.com/microsoft/syntheseus/pull/8)) ([@austint], [@fiberleif])
- Add a top-level CLI for running end-to-end search ([#26](https://github.com/microsoft/syntheseus/pull/26)) ([@kmaziarz])
- Release single-step evaluation framework and wrappers for several model types ([#14](https://github.com/microsoft/syntheseus/pull/14), [#15](https://github.com/microsoft/syntheseus/pull/15), [#20](https://github.com/microsoft/syntheseus/pull/20), [#32](https://github.com/microsoft/syntheseus/pull/32), [#35](https://github.com/microsoft/syntheseus/pull/35)) ([@kmaziarz])
- Release checkpoints for all supported single-step model types ([#21](https://github.com/microsoft/syntheseus/pull/21)) ([@kmaziarz])
- Support `*.csv` and `*.smi` formats for the single-step evaluation data ([#33](https://github.com/microsoft/syntheseus/pull/33)) ([@kmaziarz])
- Implement node evaluators commonly used in MCTS and Retro* ([#23](https://github.com/microsoft/syntheseus/pull/23), [#27](https://github.com/microsoft/syntheseus/pull/27)) ([@kmaziarz])
- Add option to terminate search when the first solution is found ([#13](https://github.com/microsoft/syntheseus/pull/13)) ([@austint])
- Add code to extract routes in order found instead of by minimum cost ([#9](https://github.com/microsoft/syntheseus/pull/9)) ([@austint])
- Declare support for type checking ([#4](https://github.com/microsoft/syntheseus/pull/4)) ([@kmaziarz])
- Add method to extract precursors from `SynthesisGraph` objects ([#36](https://github.com/microsoft/syntheseus/pull/36)) ([@austint])

### Fixed

- Fix bug where standardizing MolSetGraphs crashed ([#24](https://github.com/microsoft/syntheseus/pull/24)) ([@austint])
- Guard against rare issues in MEGAN and LocalRetro ([#29](https://github.com/microsoft/syntheseus/pull/29), [#31](https://github.com/microsoft/syntheseus/pull/31)) ([@kmaziarz])
- Change default node depth to infinity ([#16](https://github.com/microsoft/syntheseus/pull/16)) ([@austint])
- Adapt tutorials to the renaming from PR #9 ([#17](https://github.com/microsoft/syntheseus/pull/17)) ([@jagarridotorres])
- Pin `pydantic` version to `1.*` ([#10](https://github.com/microsoft/syntheseus/pull/10)) ([@kmaziarz])
- Fix compatibility with Python 3.7 ([#5](https://github.com/microsoft/syntheseus/pull/5)) ([@kmaziarz])
- Correct some typos and unclear error messages ([#39](https://github.com/microsoft/syntheseus/pull/39)) ([@austint])

## [0.1.0] - 2023-05-25

:seedling: Initial public release, containing several multi-step search algorithms and a minimal interface for single-step models.

[Unreleased]: https://github.com/microsoft/syntheseus/compare/v0.5.0...HEAD
[0.1.0]: https://github.com/microsoft/syntheseus/releases/tag/v0.1.0
[0.2.0]: https://github.com/microsoft/syntheseus/releases/tag/v0.2.0
[0.3.0]: https://github.com/microsoft/syntheseus/releases/tag/v0.3.0
[0.4.0]: https://github.com/microsoft/syntheseus/releases/tag/v0.4.0
[0.4.1]: https://github.com/microsoft/syntheseus/releases/tag/v0.4.1
[0.5.0]: https://github.com/microsoft/syntheseus/releases/tag/v0.5.0

[@austint]: https://github.com/AustinT
[@kmaziarz]: https://github.com/kmaziarz
[@jagarridotorres]: https://github.com/jagarridotorres
[@fiberleif]: https://github.com/fiberleif

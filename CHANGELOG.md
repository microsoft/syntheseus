# Changelog

All notable changes to the project are documented in this file.

The format follows [Common Changelog](https://common-changelog.org/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2023-12-19

### Added

- Add a general CLI endpoint ([#44](https://github.com/microsoft/syntheseus/pull/44)) ([@kmaziarz])
- Add support for PDVN to the search CLI ([#46](https://github.com/microsoft/syntheseus/pull/46)) ([@fiberleif])
- Add initial static documentation ([#45](https://github.com/microsoft/syntheseus/pull/45)) ([@kmaziarz])

### Changed

- Simplify single-step model setup ([#41](https://github.com/microsoft/syntheseus/pull/41), [#48](https://github.com/microsoft/syntheseus/pull/48)) ([@kmaziarz])
- Refactor single-step evaluation script and move it to cli/ ([#43](https://github.com/microsoft/syntheseus/pull/43)) ([@kmaziarz])
- Return model predictions as dataclasses instead of pydantic models ([#47](https://github.com/microsoft/syntheseus/pull/47)) ([@kmaziarz])
- Make the package compatible with PyPI ([#50](https://github.com/microsoft/syntheseus/pull/50)) ([@kmaziarz])

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

[Unreleased]: https://github.com/microsoft/syntheseus/compare/v0.3.0...HEAD
[0.1.0]: https://github.com/microsoft/syntheseus/releases/tag/v0.1.0
[0.2.0]: https://github.com/microsoft/syntheseus/releases/tag/v0.2.0
[0.3.0]: https://github.com/microsoft/syntheseus/releases/tag/v0.3.0

[@austint]: https://github.com/AustinT
[@kmaziarz]: https://github.com/kmaziarz
[@jagarridotorres]: https://github.com/jagarridotorres
[@fiberleif]: https://github.com/fiberleif

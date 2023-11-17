# Changelog

All notable changes to the project are documented in this file.

The format follows [Common Changelog](https://common-changelog.org/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Select search hyperparameters depending on which algorithm and single-step model are used ([#30](https://github.com/microsoft/syntheseus/pull/30)) ([@kmaziarz])
- Add option to override time tolerance in algorithm tests ([#25](https://github.com/microsoft/syntheseus/pull/25)) ([@austint])
- Improve the heuristic used for estimating diversity ([#22](https://github.com/microsoft/syntheseus/pull/22), [#28](https://github.com/microsoft/syntheseus/pull/28)) ([@kmaziarz])
- Improve the aesthetics of `README.md` ([#19](https://github.com/microsoft/syntheseus/pull/19)) ([@kmaziarz])

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

### Fixed

- Fix bug where standardizing MolSetGraphs crashed ([#24](https://github.com/microsoft/syntheseus/pull/24)) ([@austint])
- Guard against rare issues in MEGAN and LocalRetro ([#29](https://github.com/microsoft/syntheseus/pull/29), [#31](https://github.com/microsoft/syntheseus/pull/31)) ([@kmaziarz])
- Change default node depth to infinity ([#16](https://github.com/microsoft/syntheseus/pull/16)) ([@austint])
- Adapt tutorials to the renaming from PR #9 ([#17](https://github.com/microsoft/syntheseus/pull/17)) ([@jagarridotorres])
- Pin `pydantic` version to `1.*` ([#10](https://github.com/microsoft/syntheseus/pull/10)) ([@kmaziarz])
- Fix compatibility with Python 3.7 ([#5](https://github.com/microsoft/syntheseus/pull/5)) ([@kmaziarz])
- Pin `zipp` version to `<3.16` ([#11](https://github.com/microsoft/syntheseus/pull/11)) ([@kmaziarz])

## [0.1.0] - 2023-05-25

:seedling: Initial public release, containing several multi-step search algorithms and a minimal interface for single-step models.

[Unreleased]: https://github.com/microsoft/syntheseus/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/microsoft/syntheseus/releases/tag/v0.1.0

[@austint]: https://github.com/AustinT
[@kmaziarz]: https://github.com/kmaziarz
[@jagarridotorres]: https://github.com/jagarridotorres
[@fiberleif]: https://github.com/fiberleif

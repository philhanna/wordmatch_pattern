# Changelog

## [1.0.0] - 2026-06-17

### Added
- Ports-and-adapters (hexagonal) package structure with separate domain, application, ports, and adapters layers.
- Bigram model for improved probability calibration.
- `train` CLI (`pattern.adapters.train_cli`) to train a model from a word list.
- `match` CLI (`pattern.adapters.match_cli`) to query a trained model for a pattern.
- `train` and `match` wrapper scripts.
- Application-layer `load_model()` and re-exported `match_probability()`.
- README and source docstrings/comments.

### Changed
- Model persistence switched to joblib, with `model.joblib` as the default output of `train_cli`.
- Package moved to the project root.
- Converted all relative imports to absolute imports.

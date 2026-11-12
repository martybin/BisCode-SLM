# BisCode-SLM

A from-scratch decoder-only coding model. Weights start random. This repository does not load a pretrained language model.

## Status

The configuration system is in place. The model, tokenizer, data pipeline, trainer, and inference code are not implemented yet. No dataset has been downloaded. `download_enabled` in `configs/data/initial.yaml` is false, and a training launch stays blocked while that flag is false.

Design notes from the hardware and dataset audit:

- [docs/INITIAL_RESEARCH.md](docs/INITIAL_RESEARCH.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/DATASET_SELECTION.md](docs/DATASET_SELECTION.md)

## Layout

```text
configs/          YAML for the model, the smoke run, and the data policy
src/codeslm/      Python package
scripts/          Command-line entry points
tests/            pytest
docs/             Design and dataset research
data/             Empty until a download is approved
checkpoints/      Empty until a training run is approved
```

`scripts/download_dataset.py` is an earlier sketch. It is not the data pipeline and it has not been run.

## Environment

Use the project virtualenv, which already has PyTorch `2.15.0+cu132`:

```bash
source .venv/bin/activate
python scripts/validate_configs.py
pytest
```

`requirements.txt` lists the direct dependencies. Installing a generic `torch` wheel can replace the CUDA build already in `.venv`.

## Validate configs

`python scripts/validate_configs.py` loads:

- `configs/model/base.yaml` — 168,059,904 parameters
- `configs/model/smoke.yaml` — 40,509,952 parameters
- `configs/training/smoke_test.yaml` — 20 steps and 20,000 tokens
- `configs/data/initial.yaml` — permissive Python from CodeParrot Clean, streaming caps, downloads off

The command prints any reason a training process must refuse to start. A valid config can still be blocked.

## License

MIT. See [LICENSE](LICENSE). Dataset licenses are separate and are recorded in `docs/DATASET_SELECTION.md`.

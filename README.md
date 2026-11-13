# BisCode-SLM

A from-scratch decoder-only coding model. Weights start random. This repository does not load a pretrained language model.

## Status

The configuration system, byte-level BPE tokenizer, local data pipeline, and decoder-only model are in place. The trainer and inference code are not implemented yet. No dataset has been downloaded, and the 32,000-token tokenizer has not been trained on real code. `download_enabled` in `configs/data/initial.yaml` is false, and a training launch stays blocked while that flag is false.

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

## Tokenizer

`codeslm.tokenizer` trains a byte-level BPE model from strings you pass in. `add_prefix_space` is false, the initial alphabet is all 256 bytes, and decode preserves newlines, indentation, and special tokens. The eight special tokens from the model YAML take ids 0 through 7.

```bash
python scripts/train_tokenizer.py \
  --config configs/model/smoke.yaml \
  --corpus path/to/local_utf8_files \
  --output tokenizer/smoke \
  --vocab-size 280
```

The script reads local files only. If more than half of the requested vocabulary would be unused reserved ids, it exits without writing the file. That is the guard against training the 32,000-token vocabulary on a handful of samples.

## Data pipeline

`CodeModel` is the decoder-only network: pre-norm RMSNorm, RoPE, grouped-query attention through PyTorch SDPA, SwiGLU, and a tied embedding. `CodeModel.from_config` reads a model YAML. A CPU test checks the 168,059,904-parameter baseline.

`codeslm.data.run_pipeline` filters records already in memory. It does not download them. Permissive licenses in the data config are kept, excluded licenses are dropped, and any other license is quarantined. Empty, short, binary, generated, minified, vendored, and non-code files are excluded. Suspected secrets are quarantined, and the matched text is not written to the report. Exact SHA-256 duplicates and MinHash near-duplicates are removed. A repository or identical body that appears in validation is removed from training. `pack_texts` appends the end-of-sequence token, restarts positions on each document, and does not train a token to predict across a document boundary.

## License

MIT. See [LICENSE](LICENSE). Dataset licenses are separate and are recorded in `docs/DATASET_SELECTION.md`.

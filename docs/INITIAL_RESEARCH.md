# Initial research audit

Status: audit and design. No training code has been implemented in this step. No dataset files were downloaded. No training run was started.

Retrieval and measurement date: 2026-11-12.

This document records what was measured on the development machine, the parameter and memory arithmetic for the candidate models, and the implementation plan. Dataset-by-dataset evidence is in [DATASET_SELECTION.md](DATASET_SELECTION.md). The architecture specification is in [ARCHITECTURE.md](ARCHITECTURE.md).

## 1. Repository state

Path: `/home/marty/Desktop/projects/BisCode-SLM`

Remote: `git@github.com:martybin/BisCode-SLM.git`, branch `main`, commit `b9c49ea` ("Initial commit").

| Path | State |
| --- | --- |
| `README.md` | Title only: `# BisCode-SLM` |
| `LICENSE` | MIT, copyright 2026 Marty |
| `scripts/download_dataset.py` | Untracked. Streams `codeparrot/codeparrot-clean-train` and writes JSONL. Not executed. |
| `data/raw/` | Empty directory |
| `.venv/` | Project virtualenv, Python 3.14.4 |

The download script already targets the train repository, caps the save at 100,000 kept rows, and filters to a permissive license allowlist. It does not yet satisfy the governance requirements: it has a single skip counter, writes no provenance report, and does not record a retrieval date. It stays in the tree. The data pipeline built later should replace its role. It should not be run as-is for the baseline corpus.

## 2. Environment

Measured with the project virtualenv at `.venv/bin/python`.

| Item | Value |
| --- | --- |
| OS | Ubuntu 26.04.1 LTS |
| CPU | Intel Core i7-14650HX, 16 cores / 24 threads, max 5.2 GHz |
| RAM | 30 GiB total, about 19 GiB available at audit time, 8 GiB swap |
| Disk | 274 GB filesystem, 131 GB free |
| Python | 3.14.4 |
| PyTorch | 2.15.0+cu132, CUDA build 13.2, cuDNN 92600 |
| Other installed libraries | `transformers` 5.22.2, `datasets` 5.2.0, `tokenizers` 0.23.3, `numpy` 2.5.4, `pytest` 9.2.0, `safetensors` 0.8.0, `tensorboard` 2.21.0, `mlflow` 3.18.0, `fastapi` 0.162.5, `uvicorn` 0.54.0, `PyYAML` 6.0.3 |
| Not installed | `flash-attn`, `bitsandbytes`, `datasketch`, `docker`, `podman` |
| System `nvcc` | Not on `PATH`. The PyTorch CUDA wheel is sufficient for training. |

Python 3.14 is ahead of the versions many research libraries document. The installed PyTorch wheel runs CUDA on this machine. Later dependency pins should match this venv rather than assume a 3.10–3.12 environment.

## 3. GPU

`nvidia-smi` and a PyTorch probe were run on 2026-11-12.

| Item | Measured | Published reference |
| --- | --- | --- |
| Name | NVIDIA GeForce RTX 5070 Laptop GPU | Same |
| Architecture | Blackwell | Blackwell |
| Compute capability reported by PyTorch | 12.0 | NVIDIA laptop compare table lists 12.8 for the 50-series laptop line |
| Streaming multiprocessors | 36 | 4,608 CUDA cores on the NVIDIA 50-series laptop page (36 × 128) |
| Memory, `nvidia-smi` | 8,151 MiB total, 7,667 MiB free, display active, 41 MiB used at idle | 8 GB GDDR7 variant. The product also exists as 12 GB. This unit is 8 GB. |
| Memory visible to PyTorch | 8,082,096,128 bytes (7.527 GiB) | Driver reserves the rest |
| Memory bus | 128-bit | 128-bit |
| Specified memory bandwidth | — | 384 GB/s on the NVIDIA 50-series laptop page |
| Measured device copy traffic | 336 GB/s on a 256 MiB FP16 copy | Short benchmark, read plus write |
| Power limit | Current 77.63 W, default 55 W, max 115 W | NVIDIA lists a 50–100 W TGP range for this GPU. This OEM exposes a higher max. The audit did not change the power limit. |
| Clocks | Idle SM about 1.6 GHz. Reported max SM clock 3,090 MHz. | NVIDIA boost range 1,425–2,347 MHz |
| BF16 | `torch.cuda.is_bf16_supported()` is true. A BF16 autocast matmul succeeded. | — |
| FP16 | Autocast matmul succeeded | — |
| SDPA | Flash, memory-efficient, and math backends are all enabled | No `flash-attn` package installed |
| Display | Active. The GPU is also driving the desktop. | Training budgets must leave allocator and display headroom |

Source for the published columns: NVIDIA GeForce RTX 50 Series laptop specifications, retrieved 2026-11-12, <https://www.nvidia.com/en-us/geforce/laptops/50-series/>.

### Short GEMM benchmark

BF16 GEMMs, 10 warmup iterations and 30 timed iterations, current power limit:

| Shape | BF16 throughput |
| --- | --- |
| 4096 × 4096 × 4096 | 41.8 TFLOPS |
| (8192 × 1024) × (1024 × 1024) | 47.0 TFLOPS |
| (8192 × 1024) × (1024 × 2816) | 41.1 TFLOPS |
| (8192 × 2816) × (2816 × 1024) | 41.5 TFLOPS |

These are short bursts. A multi-hour training run at 78 W will sustain less than the burst GEMM rate because of attention, optimizer steps, launch overhead, and thermals. The planning rate used below is 12 effective TFLOPS, inside a plausible band of about 8–20 TFLOPS. Stage 3 must replace this number with a measured training step.

Practical VRAM budget for a training process: 7.0 GiB. PyTorch can see 7.53 GiB, and the desktop is using the same GPU.

## 4. Parameter count

Learned parameters, with biases off:

```text
head_dim = d_model / n_heads

per layer:
  attention = d_model^2                         # Q
            + n_kv_heads * head_dim * d_model   # K
            + n_kv_heads * head_dim * d_model   # V
            + d_model^2                         # output
  swiglu    = 3 * d_model * intermediate_size
  norms     = 2 * d_model

embedding   = vocab_size * d_model
final norm  = d_model
lm head     = 0 if tied, else vocab_size * d_model
```

`intermediate_size` for the SwiGLU width follows the usual Llama rounding: round `(8/3) * d_model` up to a multiple of 256. For `d_model = 1024` that is 2,816, which matches the requested starting point.

RoPE, causal masks, and the GQA repeat of key/value heads add no parameters.

### Recommended baseline

Tied embeddings, the requested width and depth:

| | |
| --- | --- |
| `d_model` | 1,024 |
| layers | 12 |
| query heads | 16 |
| key/value heads | 4 |
| head dimension | 64 |
| intermediate size | 2,816 |
| vocabulary | 32,000, specials included |
| context configured for training | 2,048 |
| parameters per layer | 11,274,240 |
| transformer body | 135,290,880 |
| tied embedding | 32,768,000 |
| final RMSNorm | 1,024 |
| **total** | **168,059,904 (168.06M)** |

The same model with an untied output head is 200,827,904 parameters. Both fit in 8 GB with FP32 AdamW. Tied embeddings are the baseline: the output head shares the input embedding, which removes 32.8M parameters and about 0.49 GiB of AdamW state, and 168M sits in the requested 150–200M band with margin.

### Other sizes, same rules, tied embeddings, vocab 32,000

| Name | d | L | heads | kv | intermediate | Parameters | AdamW state (16 bytes/param) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Smoke | 512 | 8 | 8 | 2 | 1,536 | 40,509,952 | 0.60 GiB |
| 50M class | 512 | 10 | 8 | 2 | 1,536 | 46,541,312 | 0.69 GiB |
| 100M class | 768 | 12 | 12 | 4 | 2,048 | 100,092,672 | 1.49 GiB |
| **Baseline** | **1024** | **12** | **16** | **4** | **2816** | **168,059,904** | **2.50 GiB** |
| 300M class | 1024 | 24 | 16 | 4 | 2,816 | 303,350,784 | 4.52 GiB |
| 470M class | 1280 | 24 | 16 | 4 | 3,584 | 469,628,160 | 7.00 GiB |

AdamW state here is FP32 master weights + FP32 gradients + FP32 first moment + FP32 second moment. It does not include activations or CUDA workspace.

The 470M configuration consumes the entire PyTorch-visible 7.53 GiB before activations. It is not a candidate for standard AdamW on this GPU. The 300M configuration is a later, tight experiment: about 4.5 GiB of optimizer state leaves roughly 2 GiB for workspace and activations, which is enough at batch 1–2 and sequence 2,048 with gradient checkpointing. It is not the first training target.

A separate tiny profile (`d_model` 64, 2 layers, vocabulary well under 32,000) belongs in the unit tests so CPU tests do not allocate a 40M model. That profile is a test fixture, not a trained model.

## 5. Training memory for the 168M baseline

Assumptions: BF16 activations, FP32 AdamW, gradient checkpointing of each transformer block, PyTorch SDPA so the score matrix is not stored.

| Component | Estimate |
| --- | --- |
| Parameters, gradients, Adam moments | 2.50 GiB |
| Checkpointed block inputs, batch 8, sequence 2,048 | 0.38 GiB |
| Peak recomputed SwiGLU activations at that batch | 0.21 GiB |
| CUDA workspace and allocator, from the GEMM probe scale | about 0.5 GiB |
| **Working estimate** | **about 3.6 GiB** |

That leaves several gigabytes inside the 7.0 GiB planning budget. Sequence length 4,096 roughly doubles the checkpoint and SwiGLU terms and is still plausible for this width. It is a later configuration. The first runs use 2,048.

Recommended first *training* micro-batch once the trainer exists: 4 sequences at length 2,048, then increase toward 8 after one measured step. Smoke tests use a smaller batch. Effective batch is made with gradient accumulation, not by filling VRAM.

Inference KV cache for one 2,048-token sequence, BF16, 12 layers, 4 key/value heads, head size 64: about 24 MiB. Local generation is not the memory constraint.

## 6. Compute budget

Training FLOPs per token, multiply-add counted as 2 FLOPs:

```text
6 * N  +  12 * n_layers * d_model * seq_len
```

The first term is the usual linear-layer forward and backward cost. The second is the attention score and attention-context matmuls. Grouped-query attention reduces key/value parameters and cache size. It does not remove those score matmuls, because every query head still attends.

For the baseline at sequence 2,048:

```text
6 * 168,059,904 + 12 * 12 * 1,024 * 2,048 = 1,310,349,312 FLOPs/token
```

| Effective TFLOPS | Tokens/s | Time for 20 × 168M tokens (3.36B) |
| ---: | ---: | ---: |
| 8 | 6,100 | 153 hours |
| 12 (planning value) | 9,200 | 102 hours |
| 20 | 15,300 | 61 hours |
| 40 (near the short GEMM peak, unlikely sustained) | 30,500 | 31 hours |

Chinchilla-scale data for this model is `20 × parameters` ≈ **3.36 billion tokens**. That is the right label for a *meaningful small-model pretraining run*. It is not a competitive coding-model budget. StarCoder2-3B was trained on hundreds of billions of tokens. This GPU is not going to reproduce that.

How to read the stages:

| Stage | Tokens | What it proves | Planning time at 9,200 tok/s |
| --- | ---: | --- | --- |
| Pipeline validation | 2 × 10^7 | Loss decreases, checkpoints resume, logs are sane | under an hour on the baseline, minutes on the smoke model |
| Experimental training | 5 × 10^8 | The data mixture and tokenizer are usable | about 15 hours |
| Meaningful baseline pretraining | 3.4 × 10^9 | A real from-scratch 168M code model, still far from frontier quality | about 4 days |
| Competitive code model | 10^11 and up | Out of scope for this machine | not this project phase |

No run in the last two rows starts without a separate approval after the pipeline has been measured.

## 7. Data conclusion

The full comparison is in [DATASET_SELECTION.md](DATASET_SELECTION.md). The decision for the first corpus:

- Train and tokenize only on `codeparrot/codeparrot-clean-train`.
- Hold out `codeparrot/codeparrot-clean-valid`. Its Hub split is named `train`. The repository id is what makes it validation data.
- Keep a row only when its `license` field is one of `apache-2.0`, `mit`, `bsd-2-clause`, `bsd-3-clause`, `isc`, `unlicense`, `cc0-1.0`.
- On a partial Datasets Server index of 477,028 train rows, those seven labels are 60.0% of rows. That index is not a full census. A streaming license count has to confirm it before a multi-day run.
- If that ratio and the indexed mean file length of 10,361 characters both generalize to the card's 5,300,000 train rows, the permissive subset is on the order of 33 GB of text and roughly 8 billion tokens at 4 characters per token. That would cover one Chinchilla pass. This is an estimate, not a measurement of the full set.
- The initial language mixture is 100% Python. The Stack v2 and the StarCoder2 file dumps are gated, the Stack v2 Hub files that were inspected contain Software Heritage identifiers rather than source text, and CodeSearchNet does not ship per-example licenses. Inventing a 70/15/10/5 multilingual mix from unavailable data would misreport the corpus.
- Instruction datasets stay out of pretraining.

Disk: 131 GB free. Leave at least 40 GB free. Do not mirror the 27.0 GiB compressed train repository or the 398 GiB Stack v2 metadata repository. The first approved pull, when it is approved, should be a stream cap of 20,000 kept permissive files (a few hundred megabytes at the observed mean size), enough to test cleaning, deduplication, and a prototype tokenizer.

## 8. What this audit did not do

- Did not download dataset shards, even a single CodeParrot gzip file.
- Did not accept any gated dataset terms.
- Did not read the Software Heritage language-model training principles document. The Stack v2 gate requires them. They have to be read before that data is used.
- Did not run a training step, so tokens/second is still an estimate.
- Did not verify, row by row, that the CodeParrot validation repositories are disjoint from the train repositories. The loader must check `repo_name` intersection on the files it actually keeps.
- Did not install packages or change GPU power limits.

## 9. Implementation plan after approval

Approval of this audit allows the code to be written. It does not allow a dataset download or a training job.

| Phase | Work | Network and GPU |
| --- | --- | --- |
| D | Package layout, YAML loader, config validation, `.gitignore` | None |
| E | Byte-level BPE wrapper, special tokens, encode/decode tests. Training the real 32k tokenizer waits for an approved corpus. Unit tests use a tiny synthetic corpus. | None |
| F | Cleaning, license quarantine, exact hash dedup, MinHash, repo-level splits, packing, provenance writers. Tests use synthetic files. `download_enabled` stays false. | None |
| G | Decoder-only model: RMSNorm, RoPE, GQA, SwiGLU, tied embeddings, SDPA causal attention | CPU tests, then one CUDA forward/backward if approved |
| H | AdamW, warmup cosine, accumulation, BF16 autocast, checkpointing, resume, TensorBoard | A 20-step smoke run only after a separate yes |
| I | Loss, perplexity, syntax checks, subprocess execution with limits. HumanEval and MBPP adapters later, test splits only. | No benchmark download until asked |
| J | Greedy and sampled generation, CLI, Python API. FastAPI stays optional. | Local checkpoint only |
| K | Rewrite the docs so they describe the code that exists | None |
| L | `pytest` on CPU, then the smoke config | The smoke config is capped at 20,000 tokens so it cannot become a long run by accident |

Stop again before:

- the first Hugging Face download,
- any training past the capped smoke config,
- enabling The Stack v2, StarCoder2 file dumps, or CodeSearchNet,
- raising sequence length to 4,096,
- starting the 300M configuration.

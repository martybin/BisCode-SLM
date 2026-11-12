# Architecture

Status: the transformer block below is still a design. `codeslm.config` loads the YAML, `codeslm.config.sizing.count_parameters` implements the parameter formula, and `codeslm.tokenizer` trains and loads the byte-level BPE tokenizer. There is no `nn.Module` yet. When the model exists, a unit test must assert that `parameters()` matches `count_parameters`. The 32,000-token tokenizer has not been trained on a real corpus.

Hardware context: NVIDIA GeForce RTX 5070 Laptop, 7.53 GiB visible to PyTorch, BF16 supported. See [INITIAL_RESEARCH.md](INITIAL_RESEARCH.md).

## Decision

Build one decoder-only transformer and select the size from YAML.

The baseline is the requested shape with **tied embeddings**:

| Field | Value |
| --- | --- |
| Parameters | 168,059,904 |
| `d_model` | 1024 |
| layers | 12 |
| query heads | 16 |
| key/value heads | 4 |
| head dimension | 64 |
| intermediate size | 2816 |
| vocabulary | 32000 |
| training context | 2048 |
| normalization | RMSNorm, pre-norm, `eps = 1e-6` |
| position encoding | RoPE, `theta = 10000` |
| attention | grouped-query, causal, PyTorch SDPA |
| feed-forward | SwiGLU |
| biases | off |
| dropout | 0.0 |
| embedding tying | on |

This shape trains on the 8 GB laptop GPU with ordinary FP32 AdamW. The memory estimate is about 3.6 GiB at micro-batch 8 and sequence 2,048 with gradient checkpointing, against a 7.0 GiB planning budget. No width or depth change is required to make the baseline fit.

An untied head would be 200,827,904 parameters and about 3.0 GiB of AdamW state. It also fits. Tying is still the baseline: it is the right default at this size, and it keeps the model inside the 150–200M band with room for activation memory.

## Block

Each layer, pre-norm:

```text
h = x + Attention(RMSNorm(x))
y = h + SwiGLU(RMSNorm(h))
```

Final RMSNorm, then a linear head tied to the input embedding.

### RMSNorm

```text
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
```

`weight` is a learned vector of size `d_model`, initialized to ones. No bias.

### Rotary embeddings

RoPE is applied to queries and keys. It is computed from position ids. There is no learned position table, so a longer context later does not require a new embedding matrix.

`rope_theta = 10000` matches a 2,048-token baseline. Raising the training context to 4,096 or beyond should be a separate experiment, and the theta should be revisited then (Code Llama used a much larger theta for long context). The first training runs do not do that.

Position ids reset at each packed document boundary. See sequence packing below.

### Grouped-query attention

16 query heads share 4 key/value heads, 4 query heads per key/value head. Head size is 64.

Projections, all bias-free:

| Matrix | Shape |
| --- | --- |
| Q | `1024 × 1024` |
| K | `1024 × 256` |
| V | `1024 × 256` |
| output | `1024 × 1024` |

Implementation path: `torch.nn.functional.scaled_dot_product_attention` with `is_causal=True` when the sequence is a single causal span. Packed sequences that contain document boundaries pass an explicit boolean mask instead, so a token cannot attend into an earlier document. SDPA's flash and memory-efficient kernels are enabled on this GPU. The `flash-attn` package is not a dependency.

Dropout inside attention is 0.

### SwiGLU

```text
SwiGLU(x) = (SiLU(x W_gate) * (x W_up)) W_down
```

`W_gate` and `W_up` are `1024 × 2816`. `W_down` is `2816 × 1024`. No biases.

2816 is `(8/3) * 1024`, rounded up to a multiple of 256. That is the standard Llama intermediate width, and it is the width requested in the project brief.

### Residual initialization

| Parameter | Initialization |
| --- | --- |
| Embeddings | normal, standard deviation 0.02 |
| Most linear layers | normal, standard deviation 0.02 |
| Attention output projection and SwiGLU down projection | normal, standard deviation `0.02 / sqrt(2 * n_layers)` |
| RMSNorm weights | ones |

The residual-output scaling is the GPT-2 scheme. It keeps the residual stream from growing with depth at initialization. Biases do not exist, so they are not initialized.

### Objective

Next-token cross-entropy. Labels are the input ids shifted by one. Pad positions are masked out of the loss. The end-of-sequence token is a real prediction target. Loss is computed in FP32.

The first pretraining run is plain causal language modeling. Fill-in-the-middle tokens are reserved in the tokenizer and unused by the loss. Using them is a later experiment.

## Tokenizer contract

The model vocabulary is the tokenizer vocabulary. Size 32,000 **includes** the eight special tokens.

| Token | Role |
| --- | --- |
| `<|pad|>` | Batch padding. Ignored by the loss. |
| `<|unk|>` | Unknown piece, if the tokenizer emits one. Byte-level BPE should rarely need it. |
| `<|bos|>` | Optional start of a document. |
| `<|eos|>` | End of a document. Predicted, and used as a document boundary while packing. |
| `<|fim_prefix|>` | Reserved. |
| `<|fim_middle|>` | Reserved. |
| `<|fim_suffix|>` | Reserved. |
| `<|sep|>` | Reserved separator. |

Algorithm: byte-level BPE trained only on the training corpus.

Requirements that matter for code:

- `add_prefix_space` is false. A leading-space convention borrowed from natural-language GPT-2 tokenizers corrupts source text.
- Newlines, indentation, and operators survive a decode unchanged. Preservation means lossless detokenization, not "one token per indent".
- The trainer sees train rows only. Validation and test text, including HumanEval and MBPP, are not tokenizer training data.

The BPE trainer's `vocab_size` is 32,000 with those specials reserved, so the embedding matrix and the tokenizer file agree.

`codeslm.tokenizer.train_tokenizer` implements this contract:

- Pretokenizer: byte-level, `add_prefix_space=false`, GPT-2 whitespace regex on. No Unicode normalizer.
- The initial alphabet is the 256 byte symbols, so unseen code still round-trips.
- Special tokens are ids 0–7 in the table order above.
- If the corpus cannot fill `vocab_size`, the remaining ids are `<|reserved_N|>`. `decode` keeps special tokens unless `skip_special_tokens=True`.
- `scripts/train_tokenizer.py` reads local UTF-8 files only. It refuses to save a tokenizer whose reserved ids are more than half of the vocabulary, which blocks an accidental 32k train on a tiny sample.

## Sequence packing

Fixed length 512, 1,024, or 2,048. 4,096 is a defined future setting and is not the default.

Packing rules:

1. Concatenate documents up to `seq_len`.
2. Insert `<|eos|>` between documents.
3. Reset RoPE position ids at each document start.
4. Build a causal mask that is also false across document boundaries.
5. Carry a trailing partial document into the next packed row instead of dropping it.
6. Mask pad positions in the loss.

A single global causal mask over packed documents is simpler and slightly wastes capacity on cross-document attention. Document-aware masks are the specified behavior because code files are independent and the extra mask is cheap at sequence 2,048.

## Precision and training numerics

Selection order, recorded in the run config:

1. CUDA and BF16 supported: BF16 autocast, FP32 master weights, no gradient scaler. This is the path on the RTX 5070 Laptop. It was checked with `torch.cuda.is_bf16_supported()` and a live BF16 matmul.
2. CUDA without BF16: FP16 autocast with a gradient scaler.
3. No CUDA: FP32.

Other training defaults for the baseline, once a training config is added:

| Knob | Baseline intent |
| --- | --- |
| Optimizer | AdamW, `beta1 = 0.9`, `beta2 = 0.95`, `eps = 1e-8` |
| Learning rate | `3e-4` peak for the 168M model, `6e-4` for the 40M smoke model |
| Schedule | Linear warmup, then cosine decay to 10% of the peak |
| Weight decay | 0.1, applied to matrices, not to norms or biases (there are no biases) |
| Gradient clip | global norm 1.0 |
| Gradient checkpointing | on for the baseline |
| Accumulation | set so a step is on the order of 0.5M tokens once the micro-batch is known |
| Seed | stored with the checkpoint |

`beta2 = 0.95` is the usual small-model language-model setting and is more stable than 0.999 at short token horizons. It is a default, not a tuned result.

## Smoke model

`configs/model/smoke.yaml` is the same architecture at a size that can overfit a toy batch and run on CPU.

| | |
| --- | --- |
| Parameters | 40,509,952 |
| `d_model` | 512 |
| layers | 8 |
| query heads | 8 |
| key/value heads | 2 |
| intermediate size | 1,536 |
| vocabulary | 32,000 |

Sharing the 32,000 vocabulary with the baseline makes the embedding a large fraction of this model (16.4M of 40.5M). That is deliberate. A smoke run should exercise the real tokenizer size. A unit-test fixture may use a tiny vocabulary. A training smoke run should not.

## Scaling ladder

These are configs to estimate, not jobs to launch together. All figures assume tied embeddings, vocabulary 32,000, FP32 AdamW, and the RTX 5070 Laptop's 7.53 GiB PyTorch memory.

| Target | Shape | Parameters | AdamW state | On this GPU |
| --- | --- | ---: | ---: | --- |
| Smoke | 512 / 8 / 8 / 2 / 1536 | 40.5M | 0.60 GiB | Comfortable |
| 50M | 512 / 10 / 8 / 2 / 1536 | 46.5M | 0.69 GiB | Comfortable |
| 100M | 768 / 12 / 12 / 4 / 2048 | 100.1M | 1.49 GiB | Comfortable |
| Baseline | 1024 / 12 / 16 / 4 / 2816 | 168.1M | 2.50 GiB | Comfortable |
| 300M | 1024 / 24 / 16 / 4 / 2816 | 303.4M | 4.52 GiB | Possible at small batch, not the first run |
| ~500M | 1280 / 24 / 16 / 4 / 3584 | 469.6M | 7.00 GiB | Does not fit standard AdamW |

A 500M experiment on this machine would need a different optimizer memory strategy (8-bit moments or CPU-offloaded states) and a new measurement. That work is out of the first implementation.

## Deferred on purpose

The module layout should leave room for these. They are not part of the first trained checkpoint:

- Fill-in-the-middle loss
- Sliding-window or linear attention
- Mixture-of-experts
- LoRA, QLoRA, and DPO
- Context lengths above 2,048
- Tensor parallel or multi-GPU. The training loop should not hard-code a single GPU in a way that forbids a later `DistributedDataParallel` wrapper, and it should not require one.

## Trade-offs

- Tied embeddings save memory and match common practice under 1B parameters. An untied head is a one-flag ablation once the trainer exists.
- GQA at a 4:1 ratio cuts key/value parameters and the generation cache. At 168M the memory savings are modest. The structure is there so later sizes inherit it.
- Dropout 0 is the pretraining default. A small dropout remains a config field for fine-tuning experiments.
- Document-boundary masks cost a custom SDPA mask and are worth it so packed files do not attend across repositories.
- BF16 is the measured path on this GPU. The FP16 and FP32 branches exist so a CPU-only test machine still runs.

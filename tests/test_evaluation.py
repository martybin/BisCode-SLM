"""Evaluation tests. No benchmark download and no training run."""

import math
import time

import torch

from codeslm.config.schema import ModelArch
from codeslm.evaluation import (
    FunctionCase,
    FunctionTest,
    check_syntax,
    execute_functions,
    perplexity,
)
from codeslm.models import CodeModel
from codeslm.training import Batch


def test_perplexity_is_the_exponential_of_token_weighted_loss() -> None:
    torch.manual_seed(0)
    model = _tiny()
    first = _batch(1)
    second = _batch(2)
    result = perplexity(model, [first, second], device=torch.device("cpu"))
    assert result.tokens == 2 * 2 * 7
    assert math.isfinite(result.loss)
    assert abs(result.perplexity - math.exp(result.loss)) < 1e-6
    model.eval()
    perplexity(model, [first])
    assert not model.training
    model.train()
    again = perplexity(model, [first])
    assert model.training
    assert again.tokens == 2 * 7


def test_python_execution_reports_pass_syntax_and_timeout() -> None:
    assert check_syntax("def add(a, b):\n    return a + b\n").ok is True
    broken = check_syntax("def add(a, b)\n    return a\n")
    assert broken.ok is False
    assert "line" in broken.error

    cases = (
        FunctionCase(
            name="add",
            source="def add(a, b):\n    return a + b\n",
            function="add",
            tests=(FunctionTest((1, 2), 3), FunctionTest((2, 2), 4)),
        ),
        FunctionCase(
            name="wrong",
            source="def add(a, b):\n    return a - b\n",
            function="add",
            tests=(FunctionTest((1, 2), 3),),
        ),
        FunctionCase(
            name="broken",
            source="def add(a, b)\n    return a + b\n",
            function="add",
            tests=(FunctionTest((1, 2), 3),),
        ),
        FunctionCase(
            name="raises",
            source="def add(a, b):\n    raise RuntimeError('nope')\n",
            function="add",
            tests=(FunctionTest((1, 2), 3),),
        ),
    )
    report = execute_functions(cases, timeout=2.0)
    by_name = {result.name: result for result in report.results}
    assert by_name["add"].status == "passed"
    assert by_name["add"].tests_passed == 2
    assert by_name["wrong"].status == "failed"
    assert by_name["broken"].status == "syntax_error"
    assert by_name["raises"].status == "runtime_error"
    assert report.pass_rate == 0.25
    assert report.syntax_error_rate == 0.25
    assert report.runtime_error_rate == 0.25
    assert report.test_pass_rate == 2 / 5

    started = time.perf_counter()
    hung = execute_functions(
        (
            FunctionCase(
                name="hang",
                source="def hang():\n    while True:\n        pass\n",
                function="hang",
                tests=(FunctionTest((), 0),),
            ),
        ),
        timeout=0.8,
    )
    elapsed = time.perf_counter() - started
    assert hung.results[0].status == "timeout"
    assert elapsed < 5.0


def _tiny() -> CodeModel:
    return CodeModel(
        ModelArch(
            architecture="decoder_only",
            vocab_size=32,
            d_model=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            intermediate_size=64,
            max_seq_len=16,
            rope_theta=10_000.0,
            rms_norm_eps=1e-6,
            tie_embeddings=True,
            bias=False,
            dropout=0.0,
            init_std=0.02,
        )
    )


def _batch(seed: int) -> Batch:
    generator = torch.Generator().manual_seed(seed)
    token_ids = torch.randint(1, 32, (2, 8), generator=generator)
    labels = token_ids.clone()
    labels[:, -1] = -100
    return Batch(input_ids=token_ids, labels=labels)

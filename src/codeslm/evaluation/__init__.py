"""Evaluation that does not train the model."""

from codeslm.evaluation.errors import EvaluationError
from codeslm.evaluation.perplexity import PerplexityResult, perplexity
from codeslm.evaluation.python_eval import (
    ExecutionReport,
    FunctionCase,
    FunctionTest,
    check_syntax,
    execute_functions,
)

__all__ = [
    "EvaluationError",
    "ExecutionReport",
    "FunctionCase",
    "FunctionTest",
    "PerplexityResult",
    "check_syntax",
    "execute_functions",
    "perplexity",
]

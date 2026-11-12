"""Parameter counts for the decoder-only architecture.

The model module does not exist yet. Training code must keep this formula and
``nn.Module.parameters()`` in agreement. Biases are included when ``bias`` is
true. RoPE and attention masks contribute no parameters.
"""


def count_parameters(
    *,
    vocab_size: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    intermediate_size: int,
    tie_embeddings: bool,
    bias: bool,
) -> int:
    """Return the number of learned parameters.

    Raises:
        ValueError: ``d_model`` is not divisible by ``n_heads``, or a size is
            not positive.
    """
    if min(vocab_size, d_model, n_layers, n_heads, n_kv_heads, intermediate_size) < 1:
        raise ValueError("architecture sizes must be positive")
    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    head_dim = d_model // n_heads
    query = d_model * d_model
    key = n_kv_heads * head_dim * d_model
    value = key
    output = d_model * d_model
    feed_forward = 3 * d_model * intermediate_size
    norms = 2 * d_model
    layer = query + key + value + output + feed_forward + norms
    if bias:
        key_value_width = n_kv_heads * head_dim
        layer += (3 * d_model) + (2 * key_value_width) + (2 * intermediate_size)
    embedding = vocab_size * d_model
    total = (n_layers * layer) + embedding + d_model
    if not tie_embeddings:
        total += embedding
    return total

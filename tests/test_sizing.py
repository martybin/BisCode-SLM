"""Parameter-count locks for the architectures in the design docs."""

from codeslm.config.sizing import count_parameters


def test_base_tied_count_matches_hand_expansion() -> None:
    d_model = 1024
    n_heads = 16
    n_kv_heads = 4
    head_dim = d_model // n_heads
    intermediate = 2816
    layers = 12
    vocab = 32000
    attention = (d_model * d_model) + (2 * n_kv_heads * head_dim * d_model) + (d_model * d_model)
    feed_forward = 3 * d_model * intermediate
    layer = attention + feed_forward + (2 * d_model)
    total = (layers * layer) + (vocab * d_model) + d_model
    assert layer == 11_274_240
    assert total == 168_059_904
    assert (
        count_parameters(
            vocab_size=vocab,
            d_model=d_model,
            n_layers=layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            intermediate_size=intermediate,
            tie_embeddings=True,
            bias=False,
        )
        == total
    )


def test_smoke_count() -> None:
    assert (
        count_parameters(
            vocab_size=32000,
            d_model=512,
            n_layers=8,
            n_heads=8,
            n_kv_heads=2,
            intermediate_size=1536,
            tie_embeddings=True,
            bias=False,
        )
        == 40_509_952
    )


def test_untied_head_adds_one_embedding_matrix() -> None:
    tied = count_parameters(
        vocab_size=32000,
        d_model=1024,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        intermediate_size=2816,
        tie_embeddings=True,
        bias=False,
    )
    untied = count_parameters(
        vocab_size=32000,
        d_model=1024,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        intermediate_size=2816,
        tie_embeddings=False,
        bias=False,
    )
    assert untied - tied == 32000 * 1024
    assert untied == 200_827_904


def test_bias_terms_are_counted_per_projection() -> None:
    common = dict(
        vocab_size=32000,
        d_model=1024,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        intermediate_size=2816,
        tie_embeddings=True,
    )
    without = count_parameters(**common, bias=False)
    with_bias = count_parameters(**common, bias=True)
    key_value_width = 4 * (1024 // 16)
    per_layer = (3 * 1024) + (2 * key_value_width) + (2 * 2816)
    assert with_bias - without == 12 * per_layer

from motif_upcycling.params import p4_layer_param_count, sarc_param_count


def test_p4_qwen25_count():
    assert 4 * p4_layer_param_count(1536, 8960, rank=8) == 2_803_724


def test_p4_qwen3_count():
    assert 4 * p4_layer_param_count(2560, 9728, rank=8) == 3_672_076


def test_sarc_params():
    assert sarc_param_count(hidden=16) == 49
    assert sarc_param_count(hidden=16, num_gates=12) == 588

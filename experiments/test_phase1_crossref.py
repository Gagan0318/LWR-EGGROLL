"""Test Phase 1 x Phase 2 cross-referencing in rank allocation.

Validates that:
  1. Without Phase 1 data, middle layers get default_mid (one step below max)
  2. With Phase 1 data, low-magnitude middle layers drop one more step
  3. High/moderate Phase 1 keeps default_mid (Phase 1 never upgrades)
  4. Most/least sensitive layers are unaffected by Phase 1
  5. RL (max_rank=4) uses correct rank steps: most=4, mid=2, low_p1=1
  6. Works for 5-layer architectures with multiple middles
  7. Phase 3 sentinel (-999) still overrides correctly

Rank set steps:
  Supervised (max_rank=8): available = [8, 4, 2, 1]
    most=8, default_mid=4, low_p1_mid=2, least=Phase 3
  Stochastic RL (max_rank=4): available = [4, 2, 1]
    most=4, default_mid=2, low_p1_mid=1, least=1
"""

from strategy_selector import select_strategy_from_dict


# ── Supervised learning (max_rank=8) ──────────────────────────

def test_supervised_no_phase1():
    """Without Phase 1, middle layers get default_mid=4."""
    rec = select_strategy_from_dict(
        {"input": 0.041, "hidden": 0.009, "output": -0.016},
    )
    assert rec.rank_allocation["input"] == 8
    assert rec.rank_allocation["hidden"] == 4   # default_mid = available[1]
    assert rec.rank_allocation["output"] == 1
    print("PASS: supervised no Phase 1 → (8, 4, 1)")


def test_supervised_low_phase1():
    """Low Phase 1 magnitude → drops one more step to rank 2."""
    rec = select_strategy_from_dict(
        {"input": 0.041, "hidden": 0.009, "output": -0.016},
        phase1_magnitudes={"input": 0.42, "hidden": 0.15, "output": 0.31},
        # median = 0.31, hidden (0.15) < median → low_p1_mid = 2
    )
    assert rec.rank_allocation["input"] == 8
    assert rec.rank_allocation["hidden"] == 2   # low Phase 1 → available[2]
    assert rec.rank_allocation["output"] == 1
    assert rec.total_budget == 11               # 8+2+1 vs 8+4+1=13
    print(f"PASS: supervised low Phase 1 → (8, 2, 1), budget={rec.total_budget}")


def test_supervised_high_phase1():
    """High Phase 1 magnitude → keeps default_mid=4. Phase 1 never upgrades."""
    rec = select_strategy_from_dict(
        {"input": 0.041, "hidden": 0.009, "output": -0.016},
        phase1_magnitudes={"input": 0.42, "hidden": 0.40, "output": 0.19},
        # median = 0.40, hidden (0.40) >= median → default_mid = 4
    )
    assert rec.rank_allocation["input"] == 8
    assert rec.rank_allocation["hidden"] == 4   # at median → keeps default
    assert rec.rank_allocation["output"] == 1
    print("PASS: supervised high Phase 1 → (8, 4, 1)")


def test_most_least_unaffected():
    """Phase 1 does NOT affect most or least sensitive layers."""
    rec = select_strategy_from_dict(
        {"input": 0.041, "hidden": 0.009, "output": -0.016},
        phase1_magnitudes={"input": 0.05, "hidden": 0.50, "output": 0.90},
        # input has lowest Phase 1 but is still most sensitive → 8
        # output has highest Phase 1 but is still least sensitive → 1
    )
    assert rec.rank_allocation["input"] == 8
    assert rec.rank_allocation["output"] == 1
    print("PASS: most/least unaffected by Phase 1")


def test_phase3_sentinel():
    """Phase 3 confirmed rank 0 (sentinel -999) overrides everything."""
    rec = select_strategy_from_dict(
        {"input": 0.041, "hidden": 0.009, "output": -999.0},
        phase1_magnitudes={"input": 0.42, "hidden": 0.15, "output": 0.31},
    )
    assert rec.rank_allocation["output"] == 0   # Phase 3 sentinel → rank 0
    assert rec.rank_allocation["hidden"] == 2   # still gets Phase 1 cross-ref
    print("PASS: Phase 3 sentinel → output=0")


# ── RL (max_rank=4): available = [4, 2, 1] ───────────────────

def test_rl_no_phase1():
    """RL without Phase 1: middle layers get default_mid=2."""
    rec = select_strategy_from_dict(
        {"input": 12.5, "hidden": 3.2, "output": -1.8},
        max_rank=4,
    )
    assert rec.rank_allocation["input"] == 4    # available[0]
    assert rec.rank_allocation["hidden"] == 2   # available[1] = default_mid
    assert rec.rank_allocation["output"] == 1
    print("PASS: RL no Phase 1 → (4, 2, 1)")


def test_rl_low_phase1():
    """RL with low Phase 1 → drops to available[2]=1."""
    rec = select_strategy_from_dict(
        {"input": 12.5, "hidden": 3.2, "output": -1.8},
        phase1_magnitudes={"input": 0.85, "hidden": 0.12, "output": 0.45},
        max_rank=4,
        # median = 0.45, hidden (0.12) < median → low_p1_mid = 1
    )
    assert rec.rank_allocation["input"] == 4
    assert rec.rank_allocation["hidden"] == 1   # low Phase 1 → available[2]
    assert rec.rank_allocation["output"] == 1
    assert rec.total_budget == 6                # 4+1+1
    print(f"PASS: RL low Phase 1 → (4, 1, 1), budget={rec.total_budget}")


def test_rl_high_phase1():
    """RL with high Phase 1 → keeps default_mid=2."""
    rec = select_strategy_from_dict(
        {"input": 12.5, "hidden": 3.2, "output": -1.8},
        phase1_magnitudes={"input": 0.85, "hidden": 0.60, "output": 0.30},
        max_rank=4,
        # median = 0.60, hidden (0.60) >= median → default_mid = 2
    )
    assert rec.rank_allocation["input"] == 4
    assert rec.rank_allocation["hidden"] == 2   # at median → keeps default
    assert rec.rank_allocation["output"] == 1
    print("PASS: RL high Phase 1 → (4, 2, 1)")


# ── Multi-layer ──────────────────────────────────────────────

def test_five_layer_supervised():
    """5-layer MLP: each middle layer independently checked."""
    rec = select_strategy_from_dict(
        {
            "input": 0.060,
            "hidden1": 0.025,
            "hidden2": 0.018,
            "hidden3": 0.010,
            "output": -0.005,
        },
        phase1_magnitudes={
            "input": 0.50,
            "hidden1": 0.35,   # above median (0.25) → default_mid=4
            "hidden2": 0.10,   # below median → low_p1_mid=2
            "hidden3": 0.25,   # at median → default_mid=4
            "output": 0.15,
        },
        # median of [0.50, 0.35, 0.10, 0.25, 0.15] = 0.25
    )
    assert rec.rank_allocation["input"] == 8     # most sensitive
    assert rec.rank_allocation["hidden1"] == 4   # high Phase 1
    assert rec.rank_allocation["hidden2"] == 2   # low Phase 1
    assert rec.rank_allocation["hidden3"] == 4   # at median → default
    assert rec.rank_allocation["output"] == 1    # least sensitive
    assert rec.total_budget == 19                # 8+4+2+4+1
    print(f"PASS: 5-layer supervised → (8,4,2,4,1), budget={rec.total_budget}")


def test_wide_hidden():
    """Wide-hidden [784,512,1024,10]: ordering input > output > hidden."""
    rec = select_strategy_from_dict(
        {"input": 0.044, "hidden": 0.001, "output": 0.012},
        phase1_magnitudes={"input": 0.38, "hidden": 0.12, "output": 0.22},
        # Phase 2 ordering: input > output > hidden
        # Middle = output, median Phase 1 = 0.22
        # output (0.22) >= median → default_mid = 4
    )
    assert rec.rank_allocation["input"] == 8
    assert rec.rank_allocation["output"] == 4    # middle, Phase 1 at median
    assert rec.rank_allocation["hidden"] == 1    # least sensitive
    print(f"PASS: wide-hidden → {rec.rank_allocation}")


if __name__ == "__main__":
    print("=" * 55)
    print("Phase 1 x Phase 2 cross-referencing tests")
    print("=" * 55)
    print()

    test_supervised_no_phase1()
    test_supervised_low_phase1()
    test_supervised_high_phase1()
    test_most_least_unaffected()
    test_phase3_sentinel()
    test_rl_no_phase1()
    test_rl_low_phase1()
    test_rl_high_phase1()
    test_five_layer_supervised()
    test_wide_hidden()

    print()
    print("=" * 55)
    print("All 10 tests passed.")
    print("=" * 55)

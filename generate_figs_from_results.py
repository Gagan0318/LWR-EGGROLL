#!/usr/bin/env python3
"""
generate_figs_from_results.py  (v2 — corrected paths)

Generates Figure 4.3 and Figure 4.7 from actual experiment result JSONs.
Run from your repo root: ~/dissertation/eggroll-diss

Usage:
    python generate_figs_from_results.py

First run with --scan to discover all seed JSONs and their configs:
    python generate_figs_from_results.py --scan
"""

import json, sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════

RESULTS = Path("results")
OUTPUT_DIR = Path(".")

# ── Pastel palette (matches existing dissertation figures) ──
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

C_VANILLA  = '#c9a8a8'   # dusty rose
C_ALIGNED  = '#8cb4c9'   # soft steel blue
C_REVERSED = '#d4b0a0'   # warm tan
C_LWR_LINE = '#4a7a94'   # darker steel blue (line plots)
C_ADV_BAR  = '#a8c9a8'   # sage green (advantage bars)


# ════════════════════════════════════════════════════════════
# SCAN MODE — discover what's on disk
# ════════════════════════════════════════════════════════════

def scan():
    """Find all seed JSONs and print dataset/config/accuracy."""
    print("Scanning all seed*.json files under results/...\n")
    found = list(RESULTS.rglob("seed*.json"))
    found.sort()
    print(f"Found {len(found)} seed files.\n")
    
    for f in found:
        try:
            with open(f) as fh:
                d = json.load(fh)
            ds = d.get("dataset", "?")
            cfg = d.get("config", "?")
            acc = d.get("best_test_acc", d.get("best_acc", "?"))
            seed = d.get("seed", "?")
            sigma = d.get("hp", {}).get("sigma", "?")
            rank = d.get("hp", {}).get("rank_spec", d.get("rank_spec_str", "?"))
            rel = f.relative_to(RESULTS)
            if isinstance(acc, float):
                acc_str = f"{acc*100:.2f}%"
            else:
                acc_str = str(acc)
            print(f"  {str(rel):60s}  ds={ds:16s} cfg={cfg:20s} seed={seed} σ={sigma} acc={acc_str}")
        except Exception as e:
            print(f"  {f.relative_to(RESULTS)}: ERROR {e}")
    
    print("\n\nDone. Use the paths above to verify the PATHS dict in this script.")


# ════════════════════════════════════════════════════════════
# SMART LOADER — search broadly for matching JSONs
# ════════════════════════════════════════════════════════════

def find_seeds(dataset, config, sigma=None, rank_spec_contains=None):
    """
    Search the entire results/ tree for seed JSONs matching the criteria.
    Returns list of best_test_acc values.
    
    Matching logic (checked inside each JSON):
      - dataset field matches (case-insensitive, underscore-flexible)
      - config field matches OR rank_spec matches
      - if sigma is specified, hp.sigma must match (within 0.001)
    """
    dataset_variants = {dataset, dataset.replace("_", "-"), dataset.replace("-", "_")}
    # Add common alternate names
    if dataset == "emnist_digits":
        dataset_variants.update({"emnist_digits", "emnist-digits", "emnist"})
    if dataset == "fashion_mnist":
        dataset_variants.update({"fashion_mnist", "fashion-mnist", "fashion"})
    
    config_variants = {config, config.replace("_", "-"), config.replace("-", "_")}
    
    matches = []
    
    for f in RESULTS.rglob("seed*.json"):
        try:
            with open(f) as fh:
                d = json.load(fh)
        except:
            continue
        
        # Check dataset
        ds = d.get("dataset", "").lower().replace("-", "_")
        if ds not in dataset_variants and dataset not in ds and ds not in dataset:
            continue
        
        # Check sigma if specified
        if sigma is not None:
            file_sigma = d.get("hp", {}).get("sigma", None)
            if file_sigma is None:
                continue
            if abs(float(file_sigma) - sigma) > 0.001:
                continue
        
        # Check config
        file_config = d.get("config", "").lower().replace("-", "_")
        config_match = any(cv.lower() in file_config or file_config in cv.lower() 
                          for cv in config_variants)
        
        # Also check rank_spec if provided
        if rank_spec_contains and not config_match:
            rs = d.get("hp", {}).get("rank_spec", {})
            rs_str = str(rs)
            if rank_spec_contains in rs_str:
                config_match = True
        
        if not config_match:
            continue
        
        # Extract accuracy
        acc = d.get("best_test_acc", d.get("best_acc", None))
        if acc is not None:
            seed = d.get("seed", "?")
            matches.append((float(acc), seed, str(f.relative_to(RESULTS))))
    
    if matches:
        matches.sort(key=lambda x: x[1])  # sort by seed
        print(f"  ✅ {dataset}/{config}" + (f" σ={sigma}" if sigma else "") + 
              f"  found {len(matches)} seeds:")
        for acc, seed, path in matches:
            print(f"       seed {seed}: {acc*100:.2f}%  ← {path}")
    
    return [m[0] for m in matches]


def mean_std_pct(accs):
    """List of accuracies (0-1) → (mean%, std%)."""
    if not accs:
        return None, None
    arr = np.array(accs) * 100
    return arr.mean(), arr.std()


# ════════════════════════════════════════════════════════════
# FIGURE 4.3 — LWR aligned (8,4,0) vs vanilla vs reversed
# ════════════════════════════════════════════════════════════

def generate_fig_4_3():
    print("\n" + "=" * 60)
    print("  FIGURE 4.3: LWR aligned (8,4,0) vs vanilla vs reversed")
    print("=" * 60)
    
    datasets = ["mnist", "fashion_mnist", "kmnist", "emnist_digits"]
    labels   = ["MNIST", "Fashion-\nMNIST", "KMNIST", "EMNIST-\nDigits"]
    
    vanilla_vals, aligned_vals, reversed_vals, advantages = [], [], [], []
    can_plot = True
    
    for ds, label in zip(datasets, labels):
        print(f"\n  --- {ds} ---")
        
        # Vanilla r=4
        v = find_seeds(ds, "vanilla_r4")
        if not v:
            v = find_seeds(ds, "eggroll_r4")
        v_mean, v_std = mean_std_pct(v)
        
        # Aligned (8,4,0)
        a = find_seeds(ds, "lwr_8_4_0")
        if not a:
            a = find_seeds(ds, "lwr_aligned", rank_spec_contains="8")
        a_mean, a_std = mean_std_pct(a)
        
        # Reversed (0,2,8)
        r = find_seeds(ds, "lwr_0_2_8")
        if not r:
            r = find_seeds(ds, "lwr_reversed")
        if not r:
            r = find_seeds(ds, "reversed")
        r_mean, r_std = mean_std_pct(r)
        
        if v_mean is None:
            print(f"  ❌ No vanilla data for {ds}")
            can_plot = False; continue
        if a_mean is None:
            print(f"  ❌ No aligned (8,4,0) data for {ds}")
            can_plot = False; continue
        
        vanilla_vals.append(v_mean)
        aligned_vals.append(a_mean)
        adv = a_mean - v_mean
        advantages.append(f"+{adv:.1f}pp")
        
        if r_mean is not None:
            reversed_vals.append(r_mean)
        else:
            reversed_vals.append(None)
            print(f"  ⚠️  No reversed data — bar will be absent")
        
        rev_str = f"{r_mean:.2f}" if r_mean is not None else "N/A"
        print(f"  SUMMARY: vanilla={v_mean:.2f}  aligned={a_mean:.2f}  "
              f"reversed={rev_str}  adv={advantages[-1]}")
    
    if not can_plot or len(vanilla_vals) < 4:
        print("\n  ❌ Cannot generate Figure 4.3 — missing data above.")
        print("     Fix the paths or run the missing experiments first.")
        return
    
    # ── Plot ──
    x = np.arange(len(labels))
    w = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    b1 = ax.bar(x - w, vanilla_vals, w, color=C_VANILLA, edgecolor='white', lw=0.8)
    b2 = ax.bar(x, aligned_vals, w, color=C_ALIGNED, edgecolor='white', lw=0.8)
    
    rev_plot = [v if v is not None else 0 for v in reversed_vals]
    has_reversed = any(v is not None for v in reversed_vals)
    b3 = ax.bar(x + w, rev_plot, w, color=C_REVERSED, edgecolor='white', lw=0.8)
    
    for j in range(len(labels)):
        ax.text(x[j], aligned_vals[j] + 1.5, advantages[j], ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#4a7a94')
    
    ax.set_ylabel('Test accuracy (%)', fontsize=12, color='#444444')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, color='#444444')
    ax.set_ylim(0, 105)
    ax.set_title('Fig 4.3  LWR aligned vs vanilla vs reversed — four datasets',
                 fontsize=14, fontweight='bold', pad=12, color='#333333')
    ax.yaxis.grid(True, alpha=0.12, color='#999999')
    ax.tick_params(colors='#666666')
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    
    legend_labels = ['Vanilla r=4', 'LWR aligned (8,4,0)', 'LWR reversed (0,2,8)']
    ax.legend([b1, b2, b3], legend_labels, loc='upper center',
              bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=10.5)
    
    plt.tight_layout()
    out = OUTPUT_DIR / "new_4.3.png"
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n  ✅ Saved: {out}")


# ════════════════════════════════════════════════════════════
# FIGURE 4.7 — σ × LWR interaction
# ════════════════════════════════════════════════════════════

def generate_fig_4_7():
    print("\n" + "=" * 60)
    print("  FIGURE 4.7: σ × LWR interaction")
    print("=" * 60)
    
    sigmas = [0.01, 0.03, 0.05, 0.10]
    
    # These files use sigma_init (not sigma) and have no dataset field.
    # Read directly from the known directory structure.
    SIGMA_DIR = RESULTS / "sigma_lwr"
    
    vanilla_means, lwr_means = [], []
    lwr_is_820 = False
    can_plot = True
    
    for sigma in sigmas:
        sigma_str = f"sigma{sigma}"
        sigma_path = SIGMA_DIR / sigma_str
        
        if not sigma_path.exists():
            # Try alternate naming
            for alt in [f"sigma{sigma:.2f}", f"sig{sigma}", f"sig{sigma:.2f}"]:
                if (SIGMA_DIR / alt).exists():
                    sigma_path = SIGMA_DIR / alt
                    break
        
        print(f"\n  --- σ = {sigma} (looking in {sigma_path.relative_to(RESULTS)}) ---")
        
        if not sigma_path.exists():
            print(f"  ❌ Directory not found: {sigma_path}")
            can_plot = False
            continue
        
        # Load vanilla
        v_accs = []
        v_dir = sigma_path / "vanilla_r4"
        if not v_dir.exists():
            v_dir = sigma_path / "eggroll_r4"
        if v_dir.exists():
            for sf in sorted(v_dir.glob("seed*.json")):
                with open(sf) as f:
                    d = json.load(f)
                acc = d.get("best_test_acc", d.get("best_acc"))
                if acc is not None:
                    v_accs.append(float(acc))
                    print(f"    vanilla seed: {float(acc)*100:.2f}%  ← {sf.relative_to(RESULTS)}")
        
        # Load LWR — prefer (8,4,0), fall back to (8,2,0)
        l_accs = []
        l_dir = sigma_path / "lwr_8_4_0"
        used_label = "(8,4,0)"
        if not l_dir.exists():
            l_dir = sigma_path / "lwr_8_2_0"
            used_label = "(8,2,0)"
            lwr_is_820 = True
        if l_dir.exists():
            for sf in sorted(l_dir.glob("seed*.json")):
                with open(sf) as f:
                    d = json.load(f)
                acc = d.get("best_test_acc", d.get("best_acc"))
                if acc is not None:
                    l_accs.append(float(acc))
                    print(f"    lwr {used_label} seed: {float(acc)*100:.2f}%  ← {sf.relative_to(RESULTS)}")
        
        v_mean, v_std = mean_std_pct(v_accs)
        l_mean, l_std = mean_std_pct(l_accs)
        
        if v_mean is None:
            print(f"  ❌ No vanilla data at σ={sigma}")
            can_plot = False
        if l_mean is None:
            print(f"  ❌ No LWR data at σ={sigma}")
            can_plot = False
        
        if v_mean is not None and l_mean is not None:
            vanilla_means.append(v_mean)
            lwr_means.append(l_mean)
            adv = l_mean - v_mean
            print(f"  SUMMARY: vanilla={v_mean:.2f}  lwr{used_label}={l_mean:.2f}  adv=+{adv:.2f}pp")
    
    if not can_plot or len(vanilla_means) < 4:
        print("\n  ❌ Cannot generate Figure 4.7 — missing data above.")
        return
    
    if lwr_is_820:
        print("\n  ⚠️  σ sweep data is (8,2,0), not (8,4,0). Legend will say (8,2,0).")
        print("     To fix: re-run σ sweep with rank_spec {(256,784):8, (256,256):4, (10,256):0}")
        lwr_label = "LWR (8,2,0)"
    else:
        lwr_label = "LWR (8,4,0)"
    
    advantages = [l - v for l, v in zip(lwr_means, vanilla_means)]
    adv_labels = [f"{a:+.2f}pp" for a in advantages]
    sigma_str_labels = [str(s) for s in sigmas]
    x = np.arange(len(sigmas))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5),
                                    gridspec_kw={'width_ratios': [1.1, 1]})
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')
    
    ax1.plot(x, vanilla_means, 'o-', color=C_VANILLA, lw=2.2, ms=8, label='Vanilla r=4')
    ax1.plot(x, lwr_means, 's-', color=C_LWR_LINE, lw=2.2, ms=8, label=lwr_label)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sigma_str_labels)
    ax1.set_xlabel('Noise scale σ', fontsize=11, color='#444444')
    ax1.set_ylabel('Test accuracy (%)', fontsize=11, color='#444444')
    ax1.set_title('Accuracy vs σ', fontsize=12, fontweight='bold', color='#333333')
    ax1.set_ylim(min(vanilla_means) - 2, max(lwr_means) + 2)
    ax1.yaxis.grid(True, alpha=0.12, color='#999999')
    ax1.legend(frameon=False, fontsize=10, loc='upper right')
    ax1.tick_params(colors='#666666')
    ax1.spines['left'].set_color('#cccccc')
    ax1.spines['bottom'].set_color('#cccccc')
    
    bars = ax2.bar(x, advantages, 0.55, color=C_ADV_BAR, edgecolor='white', lw=0.8)
    for j in range(len(sigmas)):
        offset = 0.1 if advantages[j] >= 0 else -0.1
        va = 'bottom' if advantages[j] >= 0 else 'top'
        ax2.text(x[j], advantages[j] + offset, adv_labels[j], ha='center', va=va,
                 fontsize=10, fontweight='bold', color='#5a8a5a' if advantages[j] >= 0 else '#944a4a')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sigma_str_labels)
    ax2.set_xlabel('Noise scale σ', fontsize=11, color='#444444')
    ax2.set_ylabel('LWR advantage (pp)', fontsize=11, color='#444444')
    ax2.set_title('LWR advantage by σ', fontsize=12, fontweight='bold', color='#333333')
    y_min = min(advantages) * 1.3 if min(advantages) < 0 else 0
    ax2.set_ylim(y_min, max(advantages) * 1.3)
    ax2.axhline(y=0, color='#cccccc', linewidth=0.8)
    ax2.yaxis.grid(True, alpha=0.12, color='#999999')
    ax2.tick_params(colors='#666666')
    ax2.spines['left'].set_color('#cccccc')
    ax2.spines['bottom'].set_color('#cccccc')
    
    fig.suptitle('Fig 4.7  σ × LWR interaction — allocation interacts with noise scale',
                 fontsize=14, fontweight='bold', color='#333333', y=1.02)
    
    plt.tight_layout()
    out = OUTPUT_DIR / "new_4.7.png"
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n  ✅ Saved: {out}")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--scan" in sys.argv:
        scan()
        sys.exit(0)
    
    print("=" * 60)
    print("  FIGURE GENERATOR v2 — smart JSON search")
    print(f"  Searching under: {RESULTS.resolve()}")
    print("=" * 60)
    
    generate_fig_4_3()
    generate_fig_4_7()
    
    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)

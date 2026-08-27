"""Merge all per-seed JSONs in results/variance_rank/ into summary.json.

Parses (sigma, rank) from the cell directory name (sig{s}_r{r}) rather than
requiring them in each per-seed JSON.
"""
import json
import re
from pathlib import Path
import numpy as np

ROOT = Path("results/variance_rank")
CELL_RE = re.compile(r"sig([\d.]+)_r(\d+)")


def main():
    summary = {}
    sigmas, ranks, seeds = set(), set(), set()

    for cell_dir in sorted(ROOT.iterdir()):
        if not cell_dir.is_dir():
            continue
        m = CELL_RE.match(cell_dir.name)
        if not m:
            continue
        sigma = float(m.group(1))
        rank = int(m.group(2))

        seed_files = sorted(cell_dir.glob("vsweep_*.json"))
        if not seed_files:
            continue

        accs, walls, variances = [], [], []
        n_converged = n_wallcap = 0

        for f in seed_files:
            with f.open() as fh:
                d = json.load(fh)
            # Handle both structures: fields at top level, or nested under 'result'
            payload = d.get("result", d)
            acc = payload.get("best_test_acc") or payload.get("acc") or payload.get("best")
            if acc is None:
                continue
            accs.append(float(acc))
            walls.append(float(payload.get("wall_seconds", d.get("wall_seconds", float("nan")))))
            if payload.get("variance_history"):
                variances.append(float(np.mean(payload["variance_history"])))
            elif "variance_mean" in payload:
                variances.append(float(payload["variance_mean"]))
            if payload.get("converged"):
                n_converged += 1
            if payload.get("stop_reason") == "wall_cap":
                n_wallcap += 1
            # Try to infer seed from filename
            sm = re.search(r"seed(\d+)", f.name)
            if sm:
                seeds.add(int(sm.group(1)))

        if not accs:
            continue

        sigmas.add(sigma)
        ranks.add(rank)
        summary[cell_dir.name] = {
            "sigma": sigma,
            "rank": rank,
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "wall_mean": float(np.nanmean(walls)),
            "n_seeds": len(accs),
            "n_converged": n_converged,
            "n_wallcap": n_wallcap,
            "variance_mean": float(np.mean(variances)) if variances else float("nan"),
            "variance_std": float(np.std(variances)) if variances else float("nan"),
        }

    total_wall = sum(
        v["wall_mean"] * v["n_seeds"] for v in summary.values() if not np.isnan(v["wall_mean"])
    ) / 60
    out = {
        "summary": summary,
        "sigmas": sorted(sigmas),
        "ranks": sorted(ranks),
        "seeds": sorted(seeds),
        "total_wall_minutes": total_wall,
        "architecture": [256, 256, 256],
    }
    (ROOT / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {ROOT / 'summary.json'}")
    print(f"  {len(summary)} cells, σ ∈ {sorted(sigmas)}, r ∈ {sorted(ranks)}")


if __name__ == "__main__":
    main()

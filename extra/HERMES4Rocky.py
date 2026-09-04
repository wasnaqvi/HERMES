"""HERMES4Rocky: Bayesian CMF Regression for Rocky Exoplanets."""
import argparse
import sys
sys.path.insert(0, ".")

import matplotlib.pyplot as plt
from pathlib import Path

from src.RockyUtils import load_rocky_surveys, build_pooled_dataset, build_partial_pooling_dataset
from src.Rocky import RockyModel, plot_rocky_cmf
from src.Model import ModelConfig

ALL_SURVEYS = ["Adibekyan", "Behmard", "Brewer", "Brinkman", "Ross"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="HERMES4Rocky: Bayesian CMF Regression for Rocky Exoplanets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Selective pooling examples:
  python HERMES4Rocky.py --surveys Behmard Brinkman
  python HERMES4Rocky.py --surveys Behmard Brinkman Ross
  python HERMES4Rocky.py --surveys Brinkman Ross --draws 4000 --tune 2000

Note: surveys without Star_CMF / Planet CMF data (e.g. Adibekyan, Brewer)
are silently skipped — they will contribute 0 rows to the pooled dataset.
""",
    )
    parser.add_argument(
        "--surveys", nargs="+", default=None, metavar="NAME",
        help=f"Surveys to include (default: all with CMF data). Choices: {ALL_SURVEYS}",
    )
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--star-cmf-err", type=float, default=0.03,
        help="Default stellar CMF measurement uncertainty (default: 0.03)",
    )
    parser.add_argument(
        "--planet-cmf-err", type=float, default=0.3,
        help="Default planet CMF measurement uncertainty (default: 0.3)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load data
    surveys = load_rocky_surveys()
    pooled_df = build_pooled_dataset(
        surveys,
        star_cmf_err=args.star_cmf_err,
        planet_cmf_err=args.planet_cmf_err,
        survey_subset=args.surveys,
    )
    if pooled_df.empty:
        print("ERROR: No CMF data found for the selected surveys.")
        print("Surveys with CMF pairs: Behmard (21), Brinkman (21), Ross (19)")
        sys.exit(1)

    pp_data = build_partial_pooling_dataset(pooled_df)

    included = sorted(pooled_df["survey"].unique())
    print(f"Surveys loaded: {list(surveys.keys())}")
    print(f"Surveys included in pooling: {included}")
    print(f"Pooled dataset: {len(pooled_df)} observations, {pp_data['n_planets']} unique planets")
    for sv in included:
        n = (pooled_df["survey"] == sv).sum()
        print(f"  {sv}: {n} rows")

    # 2. Fit complete pooling
    cfg = ModelConfig(draws=args.draws, tune=args.tune, target_accept=args.target_accept)
    model = RockyModel(cfg)

    print("\nFitting complete pooling model...")
    idata_pooled = model.fit_pooled(pooled_df, seed=args.seed)
    print("\n=== Complete Pooling ===")
    print(model.summarize(idata_pooled))

    # 3. Fit partial pooling
    print("\nFitting partial pooling model...")
    idata_pp = model.fit_partial_pooling(pp_data, seed=args.seed)
    print("\n=== Partial Pooling ===")
    print(model.summarize(idata_pp))

    # 4. Plot results
    label = "+".join(included)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_rocky_cmf(pooled_df, idata_pooled, title=f"Complete Pooling ({label})", ax=axes[0])
    plot_rocky_cmf(pooled_df, idata_pp, title=f"Partial Pooling ({label})", ax=axes[1])
    fig.tight_layout()

    out_path = Path("results") / "rocky_cmf_regression.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()

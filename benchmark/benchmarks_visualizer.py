"""Plot Forge benchmark CSV data.

Example:

    python benchmark/benchmarks_visualizer.py --kernel-name swiglu --metric-name speed
"""

import argparse
import json

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "data" / "all_benchmark_data.csv"
VISUALIZATIONS_PATH = Path(__file__).resolve().parent / "visualizations"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-name", required=True)
    parser.add_argument("--metric-name", required=True, choices=["speed", "memory"])
    parser.add_argument("--kernel-operation-mode", default="full", choices=["forward", "backward", "full"])
    parser.add_argument("--gpu-filter", default=None)
    parser.add_argument("--extra-config-filter", default=None)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_data(args) -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No benchmark data found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    mask = (
        (df["kernel_name"] == args.kernel_name)
        & (df["metric_name"] == args.metric_name)
        & (df["kernel_operation_mode"] == args.kernel_operation_mode)
    )
    df = df[mask].copy()

    if args.gpu_filter:
        df = df[df["gpu_name"].str.contains(args.gpu_filter, case=False, na=False)]

    if args.extra_config_filter:
        df = df[df["extra_benchmark_config_str"].str.contains(args.extra_config_filter, na=False)]

    if df.empty:
        raise ValueError("No rows match the requested filters.")

    # Keep only the most recent config when repeated benchmark runs exist.
    latest_timestamp = df["timestamp"].max()
    df = df[df["timestamp"] == latest_timestamp].copy()
    df["x_value"] = pd.to_numeric(df["x_value"], errors="ignore")
    for col in ["y_value_20", "y_value_50", "y_value_80"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def plot_data(df: pd.DataFrame, args) -> Path:
    output_path = (
        VISUALIZATIONS_PATH / f"{args.kernel_name}_{args.metric_name}_{args.kernel_operation_mode}.png"
    )
    if output_path.exists() and not args.overwrite:
        print(f"Visualization already exists: {output_path}")
        return output_path

    plt.figure(figsize=(10, 6))
    for provider, provider_df in df.groupby("kernel_provider"):
        provider_df = provider_df.sort_values("x_value")
        plt.plot(provider_df["x_value"], provider_df["y_value_50"], marker="o", label=provider)
        lower = provider_df["y_value_50"] - provider_df["y_value_20"]
        upper = provider_df["y_value_80"] - provider_df["y_value_50"]
        plt.errorbar(provider_df["x_value"], provider_df["y_value_50"], yerr=[lower, upper], fmt="none", capsize=4)

    extra_config = json.loads(df["extra_benchmark_config_str"].iloc[0])
    x_label = df["x_label"].iloc[0]
    metric_unit = df["metric_unit"].iloc[0]
    plt.title(f"{args.kernel_name} {args.metric_name} {args.kernel_operation_mode} {extra_config}")
    plt.xlabel(x_label)
    plt.ylabel(f"{args.metric_name} ({metric_unit})")
    plt.legend(title="provider")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    VISUALIZATIONS_PATH.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    if args.display:
        plt.show()
    plt.close()
    print(f"Wrote {output_path}")
    return output_path


def main():
    args = parse_args()
    plot_data(load_data(args), args)


if __name__ == "__main__":
    main()

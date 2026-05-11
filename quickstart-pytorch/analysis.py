from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from prettytable import PrettyTable


class ResultsVisualizer:
    def __init__(self) -> None:
        # runs[label] = list of per-round dicts
        self.runs: Dict[str, List[dict]] = {}

    def add_run(self, name: str, file_path: str) -> None:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {file_path}")

        with open(path, "r") as f:
            rounds = json.load(f)

        self.runs[name] = rounds
        print(f"  Loaded '{name}': {len(rounds)} rounds from {path.name}")

    def print_run_summary_table(self) -> None:
        table = PrettyTable()
        table.field_names = [
            "Run",
            "Rounds",
            "Final Train Loss",
            "Final Train Acc",
            "Final Eval Loss",
            "Final Eval Acc",
            "Final Central Loss",
            "Final Central Acc"
        ]
        table.align = "l"
        table.float_format = ".4"

        for name, rounds in self.runs.items():
            if not rounds:
                continue
            last = rounds[-1]   # final round metrics
            table.add_row([
                name,
                len(rounds),
                f"{last.get('train_loss', float('nan')):.4f}",
                f"{last.get('train_acc',  float('nan')):.4f}",
                f"{last.get('fed_eval_loss', float('nan')):.4f}",
                f"{last.get('fed_eval_acc',  float('nan')):.4f}",
                f"{last.get('central_loss',  float('nan')):.4f}",
                f"{last.get('central_acc',  float('nan')):.4f}"
            ])

        print("\n Run Summary ")
        print(table)

    def plot_metric(self, metric: str, fig_directory: str) -> None:
        fig_dir = Path(fig_directory)
        fig_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 5))

        for name, rounds in self.runs.items():
            x = [r["round"] for r in rounds]
            y = [r.get(metric, None) for r in rounds]

            # Skip if metric doesn't exist in this run
            if all(v is None for v in y):
                print(f"  Warning: metric '{metric}' not found in run '{name}', skipping.")
                continue

            plt.plot(x, y, marker="o", label=name)

        plt.title(metric.replace("_", " ").title())
        plt.xlabel("Round")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = fig_dir / f"{metric}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved plot → {out_path}")

    def plot_all(self, fig_directory: str) -> None:
        metrics = [
            "train_loss",
            "train_acc",
            "fed_eval_loss",
            "fed_eval_acc",
            "central_loss",
            "central_acc"
        ]
        print(f"\n Plotting {len(metrics)} metrics ")
        for metric in metrics:
            self.plot_metric(metric, fig_directory)

def main():
    visualizer = ResultsVisualizer()

    print("\nLoading runs")
    results_dir = Path("results/tp2_results/fedprox_vary_alpha") ## path to save results for different experiments

    available = sorted(results_dir.glob("*.json"))
    print(f"Available result files in {results_dir}/:")
    for f in available:
        print(f"  {f.name}")

    # labels = [
    #     "run_1",
    #     "run_2",
    #     "run_3",
    # ]
    # for label, path in zip(labels, available):
    #     visualizer.add_run(label, str(path))

    for path in available:
        print(f"\nLoading run from {path.name}")
        visualizer.add_run(path.stem, str(path))

    if not visualizer.runs:
        print("\nNo runs loaded")
        return

    # Summary table
    visualizer.print_run_summary_table()

    # Plots
    fig_dir = f"{results_dir}/figures"
    visualizer.plot_all(fig_dir)
    print(f"\nAll figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
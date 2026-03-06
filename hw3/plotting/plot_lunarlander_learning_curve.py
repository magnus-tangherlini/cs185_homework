import os
import sys
import glob
import csv

# Try to import matplotlib with error handling (same style as hw2/hw3 cartpole plot)
try:
    import matplotlib
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: matplotlib not available: {e}")
    print("Please install matplotlib (e.g., `uv add matplotlib` in hw3).")
    MATPLOTLIB_AVAILABLE = False
    sys.exit(1)


def load_experiment_data(exp_dir: str):
    """Load (env steps, eval return) from a hw3 LunarLander DQN experiment directory.

    Expects a `log.csv` with columns:
      - `step`: environment steps so far
      - `Eval_AverageReturn`: average eval return (only populated at eval points)
    """
    log_file = os.path.join(exp_dir, "log.csv")
    if not os.path.exists(log_file):
        return None

    envsteps: list[float] = []
    returns: list[float] = []

    try:
        with open(log_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                avg_ret_str = row.get("Eval_AverageReturn", "")
                step_str = row.get("step", "")
                if avg_ret_str == "" or step_str == "":
                    continue
                try:
                    envstep = float(step_str)
                    avg_ret = float(avg_ret_str)
                except ValueError:
                    continue
                envsteps.append(envstep)
                returns.append(avg_ret)

        if len(envsteps) == 0:
            return None

        exp_name = os.path.basename(exp_dir)
        # Clean name: strip env + seed suffix
        clean_name = exp_name.replace("LunarLander-v2_", "").split("_sd")[0]

        return {
            "name": clean_name,
            "envsteps": np.array(envsteps),
            "returns": np.array(returns),
        }
    except Exception as e:  # pragma: no cover
        print(f"Error loading {log_file}: {e}")
        import traceback

        traceback.print_exc()
        return None


def plot_learning_curves(experiments, title: str, output_file: str) -> None:
    """Plot env steps (x) vs average eval return (y) for one or more runs."""
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot create plots: matplotlib not available")
        return

    if not experiments:
        print("No experiment data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))

    for i, exp_data in enumerate(experiments):
        if exp_data is None:
            continue
        ax.plot(
            exp_data["envsteps"],
            exp_data["returns"],
            label=exp_data["name"],
            color=colors[i],
            linewidth=2,
            alpha=0.8,
        )

    ax.set_xlabel("Number of Environment Steps", fontsize=12)
    ax.set_ylabel("Average Eval Return", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    plt.close(fig)


def main() -> None:
    """Find LunarLander DQN experiments and plot eval return vs env steps."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_base_dir = os.path.join(script_dir, "..", "exp")
    exp_base_dir = os.path.abspath(exp_base_dir)

    print(f"Looking for LunarLander experiments in: {exp_base_dir}")

    if not os.path.exists(exp_base_dir):
        print(f"Experiment directory not found: {exp_base_dir}")
        return

    # Find all LunarLander DQN experiment directories
    all_exps = glob.glob(os.path.join(exp_base_dir, "LunarLander-v2_*"))
    all_exps = sorted(all_exps)

    experiments = []
    for exp_dir in all_exps:
        data = load_experiment_data(exp_dir)
        if data:
            experiments.append(data)

    print(f"Found {len(experiments)} LunarLander experiments with eval data.")

    if not experiments:
        print("No usable LunarLander logs found; nothing to plot.")
        return

    # Save plot in the hw3 root directory (parent of exp/)
    output_dir = os.path.dirname(exp_base_dir)
    output_dir = os.path.abspath(output_dir)
    output_path = os.path.join(output_dir, "lunarlander_eval_return.png")

    plot_learning_curves(
        experiments,
        title="LunarLander-v2 DQN: Eval Return vs Environment Steps",
        output_file=output_path,
    )


if __name__ == "__main__":
    main()



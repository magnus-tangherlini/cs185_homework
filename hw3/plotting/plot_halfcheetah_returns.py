import os
import sys
import glob
import csv

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
    """Load (env steps, eval return, alpha) from a HalfCheetah SAC experiment directory."""
    log_file = os.path.join(exp_dir, "log.csv")
    if not os.path.exists(log_file):
        return None

    envsteps: list[float] = []
    returns: list[float] = []
    alpha_steps: list[float] = []
    alphas: list[float] = []

    try:
        with open(log_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                step_str = row.get("step", "")
                if step_str == "":
                    continue
                try:
                    envstep = float(step_str)
                except ValueError:
                    continue

                avg_ret_str = row.get("Eval_AverageReturn", "")
                if avg_ret_str != "":
                    try:
                        envsteps.append(envstep)
                        returns.append(float(avg_ret_str))
                    except ValueError:
                        pass

                alpha_str = row.get("alpha", "")
                if alpha_str != "":
                    try:
                        alpha_steps.append(envstep)
                        alphas.append(float(alpha_str))
                    except ValueError:
                        pass

        if len(envsteps) == 0:
            return None

        exp_name = os.path.basename(exp_dir)
        clean_name = exp_name.replace("HalfCheetah-v4_", "").split("_2")[0]

        return {
            "name": clean_name,
            "envsteps": np.array(envsteps),
            "returns": np.array(returns),
            "alpha_steps": np.array(alpha_steps),
            "alphas": np.array(alphas),
        }
    except Exception as e:
        print(f"Error loading {log_file}: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_halfcheetah_comparison(fixed_data, autotune_data, output_dir: str) -> None:
    """Two separate figures: eval return comparison and alpha over training."""
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot create plots: matplotlib not available")
        return

    # --- Figure 1: Eval return for both runs ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    for data, color, label in [
        (fixed_data, "steelblue", "Fixed Temperature"),
        (autotune_data, "darkorange", "Auto-tuned Temperature"),
    ]:
        if data is not None:
            ax1.plot(
                data["envsteps"],
                data["returns"],
                label=label,
                color=color,
                linewidth=2,
                alpha=0.85,
            )

    ax1.set_xlabel("Number of Environment Steps", fontsize=12)
    ax1.set_ylabel("Average Eval Return", fontsize=12)
    ax1.set_title("HalfCheetah-v4: Eval Return", fontsize=14, fontweight="bold")
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    out1 = os.path.join(output_dir, "halfcheetah_sac_eval_return.png")
    plt.savefig(out1, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {out1}")
    plt.close(fig1)

    # --- Figure 2: Alpha (temperature) over training for auto-tuned run ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    if autotune_data is not None and len(autotune_data["alphas"]) > 0:
        ax2.plot(
            autotune_data["alpha_steps"],
            autotune_data["alphas"],
            color="darkorange",
            linewidth=2,
            alpha=0.85,
        )
    ax2.set_xlabel("Number of Environment Steps", fontsize=12)
    ax2.set_ylabel("Temperature (α)", fontsize=12)
    ax2.set_title("HalfCheetah-v4: Alpha (Auto-tuned)", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = os.path.join(output_dir, "halfcheetah_sac_alpha.png")
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {out2}")
    plt.close(fig2)


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_base_dir = os.path.abspath(os.path.join(script_dir, "..", "exp"))

    print(f"Looking for HalfCheetah SAC experiments in: {exp_base_dir}")

    if not os.path.exists(exp_base_dir):
        print(f"Experiment directory not found: {exp_base_dir}")
        return

    fixed_dirs = sorted(glob.glob(os.path.join(exp_base_dir, "HalfCheetah-v4_sac_sd1_*")))
    autotune_dirs = sorted(glob.glob(os.path.join(exp_base_dir, "HalfCheetah-v4_sac_autotune_sd1_*")))

    fixed_data = None
    for d in fixed_dirs:
        data = load_experiment_data(d)
        if data:
            fixed_data = data
            print(f"Loaded fixed-temp data from: {os.path.basename(d)} ({len(data['envsteps'])} eval points)")
            break

    autotune_data = None
    for d in autotune_dirs:
        data = load_experiment_data(d)
        if data:
            autotune_data = data
            print(f"Loaded autotune data from: {os.path.basename(d)} ({len(data['envsteps'])} eval points, {len(data['alphas'])} alpha points)")
            break

    if fixed_data is None and autotune_data is None:
        print("No usable HalfCheetah logs found; nothing to plot.")
        return

    output_dir = os.path.dirname(exp_base_dir)
    plot_halfcheetah_comparison(fixed_data, autotune_data, output_dir)


if __name__ == "__main__":
    main()

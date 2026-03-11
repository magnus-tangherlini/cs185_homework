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
    MATPLOTLIB_AVAILABLE = False
    sys.exit(1)


def load_experiment_data(exp_dir: str):
    log_file = os.path.join(exp_dir, "log.csv")
    if not os.path.exists(log_file):
        return None

    eval_steps, eval_returns = [], []
    q_steps, q_values = [], []

    try:
        with open(log_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                step_str = row.get("step", "")
                if step_str == "":
                    continue
                try:
                    step = float(step_str)
                except ValueError:
                    continue

                ret_str = row.get("Eval_AverageReturn", "")
                if ret_str != "":
                    try:
                        eval_steps.append(step)
                        eval_returns.append(float(ret_str))
                    except ValueError:
                        pass

                q_str = row.get("q_values", "")
                if q_str != "":
                    try:
                        q_steps.append(step)
                        q_values.append(float(q_str))
                    except ValueError:
                        pass

        if not eval_steps:
            return None

        exp_name = os.path.basename(exp_dir)
        # e.g. Hopper-v4_sac_clipq_sd1 -> sac_clipq
        clean_name = exp_name.replace("Hopper-v4_", "").split("_sd")[0]

        return {
            "name": clean_name,
            "eval_steps": np.array(eval_steps),
            "eval_returns": np.array(eval_returns),
            "q_steps": np.array(q_steps),
            "q_values": np.array(q_values),
        }
    except Exception as e:
        print(f"Error loading {log_file}: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_hopper(experiments, output_dir: str) -> None:
    if not MATPLOTLIB_AVAILABLE:
        return
    if not experiments:
        print("No experiment data to plot.")
        return

    colors = ["steelblue", "darkorange", "seagreen", "crimson"]

    # --- Figure 1: Eval return ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for i, exp in enumerate(experiments):
        ax1.plot(exp["eval_steps"], exp["eval_returns"],
                 label=exp["name"], color=colors[i % len(colors)],
                 linewidth=2, alpha=0.85)
    ax1.set_xlabel("Number of Environment Steps", fontsize=12)
    ax1.set_ylabel("Average Eval Return", fontsize=12)
    ax1.set_title("Hopper-v4: Eval Return", fontsize=14, fontweight="bold")
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    out1 = os.path.join(output_dir, "hopper_sac_eval_return.png")
    plt.savefig(out1, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {out1}")
    plt.close(fig1)

    # --- Figure 2: Q values ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for i, exp in enumerate(experiments):
        if len(exp["q_values"]) > 0:
            ax2.plot(exp["q_steps"], exp["q_values"],
                     label=exp["name"], color=colors[i % len(colors)],
                     linewidth=2, alpha=0.85)
    ax2.set_xlabel("Number of Environment Steps", fontsize=12)
    ax2.set_ylabel("Q Values", fontsize=12)
    ax2.set_title("Hopper-v4: Q Values", fontsize=14, fontweight="bold")
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = os.path.join(output_dir, "hopper_sac_q_values.png")
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {out2}")
    plt.close(fig2)


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_base_dir = os.path.abspath(os.path.join(script_dir, "..", "exp"))

    print(f"Looking for Hopper SAC experiments in: {exp_base_dir}")
    if not os.path.exists(exp_base_dir):
        print(f"Experiment directory not found: {exp_base_dir}")
        return

    exp_dirs = sorted(glob.glob(os.path.join(exp_base_dir, "Hopper-v4_sac_*")))

    experiments = []
    for d in exp_dirs:
        data = load_experiment_data(d)
        if data:
            experiments.append(data)
            print(f"Loaded: {os.path.basename(d)} ({len(data['eval_steps'])} eval points, {len(data['q_values'])} q points)")

    if not experiments:
        print("No usable Hopper logs found.")
        return

    output_dir = os.path.dirname(exp_base_dir)
    plot_hopper(experiments, output_dir)


if __name__ == "__main__":
    main()

"""Plot training curves comparing MSE and Flow policies."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_training_data(exp_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Load training loss from CSV and eval metrics from WandB summary.
    
    Returns:
        train_df: DataFrame with 'step' and 'train/loss' columns
        eval_data: Dict with eval metrics (from WandB summary)
    """
    csv_path = exp_dir / "log.csv"
    train_df = pd.read_csv(csv_path)
    
    # Try to load eval metrics from WandB summary
    eval_data = {}
    wandb_summary_path = exp_dir / "wandb" / "files" / "wandb-summary.json"
    if wandb_summary_path.exists():
        with open(wandb_summary_path) as f:
            summary = json.load(f)
            # Extract eval metrics if available
            for key, value in summary.items():
                if key.startswith("eval/"):
                    eval_data[key] = value
    
    return train_df, eval_data


def load_eval_from_wandb_runs(exp_dir: Path) -> pd.DataFrame:
    """Load eval metrics from WandB run files.
    
    This reads the actual run data to get eval/mean_reward at each step.
    Tries multiple methods:
    1. WandB API (if logged in)
    2. Reading local .wandb file using wandb internals
    """
    import wandb
    
    wandb_dir = exp_dir / "wandb"
    
    # Method 1: Try WandB API
    # Extract run ID from directory name (e.g., run-omcpimzq.wandb -> omcpimzq)
    run_files = list(wandb_dir.glob("run-*.wandb"))
    if run_files:
        run_id = run_files[0].stem.replace("run-", "")
        project = "hw1-imitation"  # Default project name
        
        try:
            api = wandb.Api()
            
            # Try to get entity from wandb settings, or try without entity
            # WandB API will use default entity if not specified
            try:
                # First try: use default entity (from wandb settings)
                run = api.run(f"{project}/{run_id}")
                history = run.history(keys=["eval/mean_reward", "step"])
                if not history.empty:
                    print(f"  Loaded eval history from WandB API: {len(history)} points")
                    return history
            except Exception:
                # If that fails, the run might not be synced or entity is different
                # Try to find the entity by searching
                pass
            
            # Alternative: try to get entity from email in metadata
            metadata_path = wandb_dir / "files" / "wandb-metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    email = metadata.get("email", "")
                    # Try common entity patterns from email
                    if "@" in email:
                        possible_entity = email.split("@")[0]
                        try:
                            run = api.run(f"{possible_entity}/{project}/{run_id}")
                            history = run.history(keys=["eval/mean_reward", "step"])
                            if not history.empty:
                                print(f"  Loaded eval history from WandB API: {len(history)} points")
                                return history
                        except Exception:
                            pass
        except Exception as e:
            print(f"  WandB API not available: {e}")
    
    # Method 2: Try reading local .wandb file using wandb's internal tools
    if run_files:
        try:
            # Use wandb's internal file reading
            import wandb.sdk.wandb_run as wandb_run
            from wandb.sdk.internal.file_stream import FileStream
            
            run_file = run_files[0]
            # This is more complex - let's try a simpler approach
            # Read the binary file and try to extract data
            # Actually, wandb doesn't expose easy local file reading
            
            # Alternative: Try to use wandb's offline sync or read directly
            # But this is complex, so we'll just return empty and use summary
            pass
        except Exception:
            pass
    
    print(f"  Could not load eval history - will use final value from summary")
    return pd.DataFrame()


def plot_training_loss(ax, train_data, label, color=None, xlim=None, ylim=None):
    """Plot training loss on given axis."""
    ax.plot(train_data["step"], train_data["train/loss"], 
            label=label, linewidth=2, alpha=0.8, color=color)
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title(f"Training Loss - {label}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)


def plot_eval_reward(ax, eval_data, eval_summary, label, color=None, marker="o", xlim=None, ylim=None):
    """Plot evaluation reward on given axis."""
    has_history = not eval_data.empty and "eval/mean_reward" in eval_data.columns
    
    if has_history:
        ax.plot(eval_data["step"], eval_data["eval/mean_reward"], 
                label=label, linewidth=2, marker=marker, markersize=6, alpha=0.8, color=color)
    elif "eval/mean_reward" in eval_summary:
        ax.axhline(y=eval_summary["eval/mean_reward"], 
                  label=f"{label} (final)", linestyle="--", linewidth=2, alpha=0.8, color=color)
    
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Mean Reward", fontsize=12)
    ax.set_title(f"Evaluation Mean Reward - {label}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)


def generate_all_plots(mse_dir: Path, flow_dir: Path, output_dir: Path | None = None):
    """Generate all 6 plots: individual MSE, individual Flow, and combined comparisons.
    
    Args:
        mse_dir: Path to MSE experiment directory
        flow_dir: Path to Flow experiment directory
        output_dir: Directory to save plots (default: current directory)
    """
    print(f"Loading data from {mse_dir} and {flow_dir}...")
    
    # Load training data
    mse_train, mse_eval_summary = load_training_data(mse_dir)
    flow_train, flow_eval_summary = load_training_data(flow_dir)
    
    print("Loading eval history (this requires WandB API if logged in)...")
    # Load eval data
    mse_eval = load_eval_from_wandb_runs(mse_dir)
    flow_eval = load_eval_from_wandb_runs(flow_dir)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(".")
    
    # Colors for consistency
    mse_color = "#1f77b4"  # Blue
    flow_color = "#ff7f0e"  # Orange
    
    # Calculate consistent axis limits
    # X-axis: training steps (use max of both)
    max_step = max(mse_train["step"].max(), flow_train["step"].max())
    min_step = min(mse_train["step"].min(), flow_train["step"].min())
    x_padding = (max_step - min_step) * 0.02  # 2% padding
    
    # Y-axis for training loss: use global min/max
    loss_min = min(mse_train["train/loss"].min(), flow_train["train/loss"].min())
    loss_max = max(mse_train["train/loss"].max(), flow_train["train/loss"].max())
    
    # Y-axis for eval reward: use global min/max if available
    eval_min = None
    eval_max = None
    if not mse_eval.empty and "eval/mean_reward" in mse_eval.columns:
        eval_min = mse_eval["eval/mean_reward"].min()
        eval_max = mse_eval["eval/mean_reward"].max()
    if not flow_eval.empty and "eval/mean_reward" in flow_eval.columns:
        if eval_min is None:
            eval_min = flow_eval["eval/mean_reward"].min()
            eval_max = flow_eval["eval/mean_reward"].max()
        else:
            eval_min = min(eval_min, flow_eval["eval/mean_reward"].min())
            eval_max = max(eval_max, flow_eval["eval/mean_reward"].max())
    
    # Add padding to eval range
    if eval_min is not None and eval_max is not None:
        eval_range = eval_max - eval_min
        eval_padding = eval_range * 0.1 if eval_range > 0 else 0.1
        eval_min = max(0, eval_min - eval_padding)  # Don't go below 0 for reward
        eval_max = eval_max + eval_padding
    
    # Set axis limits
    x_lim = (min_step - x_padding, max_step + x_padding)
    loss_y_lim = (loss_min * 0.8, loss_max * 1.2)  # For log scale, use multipliers
    eval_y_lim = (eval_min, eval_max) if eval_min is not None and eval_max is not None else None
    
    # 1. MSE Individual - Training Loss
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plot_training_loss(ax, mse_train, "MSE Policy", color=mse_color, xlim=x_lim, ylim=loss_y_lim)
    plt.tight_layout()
    mse_loss_path = output_dir / "mse_training_loss.png"
    plt.savefig(mse_loss_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {mse_loss_path}")
    
    # 2. MSE Individual - Eval Reward
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plot_eval_reward(ax, mse_eval, mse_eval_summary, "MSE Policy", color=mse_color, marker="o", 
                     xlim=x_lim, ylim=eval_y_lim)
    plt.tight_layout()
    mse_eval_path = output_dir / "mse_eval_reward.png"
    plt.savefig(mse_eval_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {mse_eval_path}")
    
    # 3. Flow Individual - Training Loss
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plot_training_loss(ax, flow_train, "Flow Policy", color=flow_color, xlim=x_lim, ylim=loss_y_lim)
    plt.tight_layout()
    flow_loss_path = output_dir / "flow_training_loss.png"
    plt.savefig(flow_loss_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {flow_loss_path}")
    
    # 4. Flow Individual - Eval Reward
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plot_eval_reward(ax, flow_eval, flow_eval_summary, "Flow Policy", color=flow_color, marker="s",
                     xlim=x_lim, ylim=eval_y_lim)
    plt.tight_layout()
    flow_eval_path = output_dir / "flow_eval_reward.png"
    plt.savefig(flow_eval_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {flow_eval_path}")
    
    # 5. Combined - Training Loss Comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(mse_train["step"], mse_train["train/loss"], 
            label="MSE Policy", linewidth=2, alpha=0.8, color=mse_color)
    ax.plot(flow_train["step"], flow_train["train/loss"], 
            label="Flow Policy", linewidth=2, alpha=0.8, color=flow_color)
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title("Training Loss Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    ax.set_xlim(x_lim)
    ax.set_ylim(loss_y_lim)
    plt.tight_layout()
    combined_loss_path = output_dir / "combined_training_loss.png"
    plt.savefig(combined_loss_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {combined_loss_path}")
    
    # 6. Combined - Eval Reward Comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    has_mse_eval_history = not mse_eval.empty and "eval/mean_reward" in mse_eval.columns
    has_flow_eval_history = not flow_eval.empty and "eval/mean_reward" in flow_eval.columns
    
    if has_mse_eval_history:
        ax.plot(mse_eval["step"], mse_eval["eval/mean_reward"], 
                label="MSE Policy", linewidth=2, marker="o", markersize=6, alpha=0.8, color=mse_color)
    elif "eval/mean_reward" in mse_eval_summary:
        ax.axhline(y=mse_eval_summary["eval/mean_reward"], 
                  label="MSE Policy (final)", linestyle="--", linewidth=2, alpha=0.8, color=mse_color)
    
    if has_flow_eval_history:
        ax.plot(flow_eval["step"], flow_eval["eval/mean_reward"], 
                label="Flow Policy", linewidth=2, marker="s", markersize=6, alpha=0.8, color=flow_color)
    elif "eval/mean_reward" in flow_eval_summary:
        ax.axhline(y=flow_eval_summary["eval/mean_reward"], 
                  label="Flow Policy (final)", linestyle="--", linewidth=2, alpha=0.8, color=flow_color)
    
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Mean Reward", fontsize=12)
    ax.set_title("Evaluation Mean Reward Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_lim)
    if eval_y_lim:
        ax.set_ylim(eval_y_lim)
    plt.tight_layout()
    combined_eval_path = output_dir / "combined_eval_reward.png"
    plt.savefig(combined_eval_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {combined_eval_path}")
    
    print(f"\n✓ Generated 6 plots in {output_dir}")


def plot_single_policy(exp_dir: Path, policy_name: str, output_path: Path | None = None):
    """Generate a combined plot (training loss + eval reward) for a single policy.
    
    Args:
        exp_dir: Path to experiment directory (e.g., exp/mse)
        policy_name: Name of the policy (e.g., "MSE" or "Flow")
        output_path: Path to save the plot
    """
    print(f"Loading data from {exp_dir}...")
    
    # Load training data
    train_data, eval_summary = load_training_data(exp_dir)
    
    print("Loading eval history (this requires WandB API if logged in)...")
    # Load eval data
    eval_data = load_eval_from_wandb_runs(exp_dir)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training Loss
    ax1 = axes[0]
    plot_training_loss(ax1, train_data, f"{policy_name} Policy")
    
    # Plot 2: Evaluation Reward
    ax2 = axes[1]
    has_history = not eval_data.empty and "eval/mean_reward" in eval_data.columns
    marker = "o" if policy_name == "MSE" else "s"
    color = "#1f77b4" if policy_name == "MSE" else "#ff7f0e"
    plot_eval_reward(ax2, eval_data, eval_summary, f"{policy_name} Policy", color=color, marker=marker)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comparison(mse_dir: Path, flow_dir: Path, output_path: Path | None = None):
    """Plot comparison of MSE and Flow policies (legacy function for backward compatibility).
    
    Args:
        mse_dir: Path to MSE experiment directory (e.g., exp/mse)
        flow_dir: Path to Flow experiment directory (e.g., exp/flow)
        output_path: Optional path to save the plot
    """
    output_dir = output_path.parent if output_path else Path("plots")
    generate_all_plots(mse_dir, flow_dir, output_dir)


def main():
    """Main function to generate plots."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot training curves for MSE and Flow policies")
    parser.add_argument(
        "--mse-dir",
        type=Path,
        default=Path("exp/mse"),
        help="Path to MSE experiment directory (default: exp/mse)",
    )
    parser.add_argument(
        "--flow-dir",
        type=Path,
        default=Path("exp/flow"),
        help="Path to Flow experiment directory (default: exp/flow)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Output directory for all plots (default: plots/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="[Deprecated] Single output path (use --output-dir instead)",
    )
    parser.add_argument(
        "--single",
        type=str,
        choices=["mse", "flow"],
        default=None,
        help="Generate a single combined plot for one policy only (mse or flow)",
    )
    parser.add_argument(
        "--single-dir",
        type=Path,
        default=None,
        help="Directory for single policy plot (default: exp/{mse|flow})",
    )
    
    args = parser.parse_args()
    
    # Handle single policy plot
    if args.single:
        if args.single_dir:
            exp_dir = args.single_dir
        else:
            exp_dir = Path(f"exp/{args.single}")
        
        if not exp_dir.exists():
            print(f"Error: Directory not found: {exp_dir}")
            return
        
        output_path = args.output if args.output else Path(f"plots/{args.single}_combined.png")
        plot_single_policy(exp_dir, args.single.upper(), output_path)
        return
    
    # Handle comparison plots
    if not args.mse_dir.exists():
        print(f"Error: MSE directory not found: {args.mse_dir}")
        return
    
    if not args.flow_dir.exists():
        print(f"Error: Flow directory not found: {args.flow_dir}")
        return
    
    # Use output-dir, or fall back to output's parent if provided (for backward compatibility)
    if args.output is not None:
        output_dir = args.output.parent
    else:
        output_dir = args.output_dir
    
    generate_all_plots(args.mse_dir, args.flow_dir, output_dir)


if __name__ == "__main__":
    main()


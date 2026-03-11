import os
import sys
import glob
import csv

# Try to import matplotlib with error handling (same style as hw2)
try:
    import matplotlib
    # Use non-interactive backend
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
    """Load training and eval returns from a MsPacman experiment directory.
    
    Returns:
        dict with 'train_envsteps', 'train_returns', 'eval_envsteps', 'eval_returns'
    """
    log_file = os.path.join(exp_dir, "log.csv")
    if not os.path.exists(log_file):
        return None

    train_envsteps: list[float] = []
    train_returns: list[float] = []
    eval_envsteps: list[float] = []
    eval_returns: list[float] = []

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
                
                # Check for eval return
                eval_ret_str = row.get("Eval_AverageReturn", "")
                if eval_ret_str != "":
                    try:
                        eval_ret = float(eval_ret_str)
                        eval_envsteps.append(envstep)
                        eval_returns.append(eval_ret)
                    except ValueError:
                        pass
                
                # Check for training return
                train_ret_str = row.get("Train_EpisodeReturn", "")
                if train_ret_str != "":
                    try:
                        train_ret = float(train_ret_str)
                        train_envsteps.append(envstep)
                        train_returns.append(train_ret)
                    except ValueError:
                        pass
        
        if len(eval_envsteps) == 0 and len(train_envsteps) == 0:
            return None
        
        return {
            'train_envsteps': np.array(train_envsteps),
            'train_returns': np.array(train_returns),
            'eval_envsteps': np.array(eval_envsteps),
            'eval_returns': np.array(eval_returns),
        }
    except Exception as e:
        print(f"Error loading {log_file}: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_returns(exp_data, title, output_file):
    """Plot training and eval returns on the same axes."""
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot create plots: matplotlib not available")
        return
    
    if exp_data is None:
        print("No data to plot")
        return
    
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot training returns
        if len(exp_data['train_envsteps']) > 0:
            ax.plot(
                exp_data['train_envsteps'],
                exp_data['train_returns'],
                label='Training Return',
                color='blue',
                linewidth=1.5,
                alpha=0.7,
                marker='.',
                markersize=2
            )
        
        # Plot eval returns
        if len(exp_data['eval_envsteps']) > 0:
            ax.plot(
                exp_data['eval_envsteps'],
                exp_data['eval_returns'],
                label='Eval Return',
                color='red',
                linewidth=2,
                alpha=0.9,
                marker='o',
                markersize=6
            )
        
        ax.set_xlabel('Number of Environment Steps', fontsize=12)
        ax.set_ylabel('Average Return', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
        plt.close(fig)
    except Exception as e:
        print(f"Error creating plot: {e}")
        import traceback
        traceback.print_exc()


def main():
    # Get the experiment directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_base_dir = os.path.join(script_dir, '..', 'exp')
    exp_base_dir = os.path.abspath(exp_base_dir)
    
    print(f"Looking for MsPacman experiments in: {exp_base_dir}")
    
    if not os.path.exists(exp_base_dir):
        print(f"Experiment directory not found: {exp_base_dir}")
        return
    
    # Find all MsPacman experiment directories
    all_exps = glob.glob(os.path.join(exp_base_dir, 'MsPacman*'))
    
    if len(all_exps) == 0:
        print("No MsPacman experiments found!")
        return
    
    # Load data from first experiment (or combine multiple if needed)
    exp_data = None
    for exp_dir in sorted(all_exps):
        data = load_experiment_data(exp_dir)
        if data:
            exp_data = data
            print(f"Loaded data from: {os.path.basename(exp_dir)}")
            print(f"  Training episodes: {len(data['train_envsteps'])}")
            print(f"  Eval points: {len(data['eval_envsteps'])}")
            break
    
    if exp_data is None:
        print("No valid experiment data found!")
        return
    
    # Create output directory (same as exp_base_dir parent)
    output_dir = os.path.dirname(exp_base_dir)
    output_dir = os.path.abspath(output_dir)
    print(f"Saving plot to: {output_dir}")
    
    # Plot training and eval returns
    plot_returns(
        exp_data,
        'MsPacman: Training vs Eval Returns',
        os.path.join(output_dir, 'mspacman_training_vs_eval_returns.png')
    )
    
    # Print explanation
    print("\n" + "="*70)
    print("EXPLANATION: Why Training and Eval Returns Look Different Early in Training")
    print("="*70)
    print("""
1. **Training Return** (blue line):
   - Collected during training episodes with exploration
   - Uses epsilon-greedy policy (high epsilon early = lots of random actions)
   - Agent is actively exploring and learning
   - Performance is lower because it's trading off exploration vs exploitation
   - Episodes may terminate early due to poor actions during exploration

2. **Eval Return** (red line):
   - Collected during evaluation episodes (typically with epsilon=0 or very low)
   - Uses the current greedy policy (best actions according to Q-network)
   - No exploration - agent acts optimally according to its current knowledge
   - Performance reflects what the agent has learned so far
   - Can be higher than training return because it's not exploring randomly

3. **Why they differ early in training**:
   - Early in training, the Q-network is poorly initialized
   - Training episodes: High epsilon → many random actions → lower returns
   - Eval episodes: Greedy policy → uses learned Q-values → may perform better
     than random exploration, but still poor because Q-network hasn't learned much
   - As training progresses, both should converge as the Q-network improves
   - The gap narrows as epsilon decays and the agent learns better policies

4. **Key Insight**:
   - Training return shows the cost of exploration during learning
   - Eval return shows the true performance of the learned policy
   - The difference is the "exploration cost" - what you pay to learn
    """)


if __name__ == "__main__":
    main()


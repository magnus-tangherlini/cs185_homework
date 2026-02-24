import os
import sys
import glob
import csv

# Try to import matplotlib with error handling
try:
    import matplotlib
    # Try Agg backend first (non-interactive)
    try:
        matplotlib.use('Agg')
    except:
        pass
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: matplotlib not available: {e}")
    print("Please install matplotlib: pip install matplotlib")
    MATPLOTLIB_AVAILABLE = False
    sys.exit(1)

def load_experiment_data(exp_dir):
    """Load log data from an experiment directory."""
    log_file = os.path.join(exp_dir, 'log.csv')
    if not os.path.exists(log_file):
        return None
    
    try:
        envsteps = []
        returns = []
        
        with open(log_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    envstep = float(row['Train_EnvstepsSoFar'])
                    return_val = float(row['Eval_AverageReturn'])
                    envsteps.append(envstep)
                    returns.append(return_val)
                except (ValueError, KeyError) as e:
                    continue
        
        if len(envsteps) == 0:
            return None
        
        # Extract the experiment name from the directory
        exp_name = os.path.basename(exp_dir)
        # Clean up the name for the legend (remove timestamp and common prefixes)
        clean_name = exp_name.replace('CartPole-v0_cartpole_', '').replace('_sd1_', '_').split('_2026')[0]
        
        return {
            'name': clean_name,
            'envsteps': np.array(envsteps),
            'returns': np.array(returns),
        }
    except Exception as e:
        print(f"Error loading {log_file}: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_learning_curves(experiments, title, output_file):
    """Plot learning curves for multiple experiments."""
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot create plots: matplotlib not available")
        return
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
        
        for i, exp_data in enumerate(experiments):
            if exp_data is None:
                continue
            ax.plot(
                exp_data['envsteps'],
                exp_data['returns'],
                label=exp_data['name'],
                color=colors[i],
                linewidth=2,
                alpha=0.8
            )
        
        ax.set_xlabel('Number of Environment Steps', fontsize=12)
        ax.set_ylabel('Average Return', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
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
    # Get the experiment directory - go up from scripts/ to src/ to hw2/ to exp/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_base_dir = os.path.join(script_dir, '..', '..', 'exp')
    exp_base_dir = os.path.abspath(exp_base_dir)
    
    print(f"Looking for experiments in: {exp_base_dir}")
    
    if not os.path.exists(exp_base_dir):
        print(f"Experiment directory not found: {exp_base_dir}")
        return
    
    # Find all experiment directories
    all_exps = glob.glob(os.path.join(exp_base_dir, 'CartPole-v0_cartpole*'))
    
    # Separate into small batch (without "lb") and large batch (with "lb")
    small_batch_exps = []
    large_batch_exps = []
    
    for exp_dir in sorted(all_exps):
        exp_name = os.path.basename(exp_dir)
        # Check if it's a large batch experiment (contains "_lb_")
        if '_lb_' in exp_name:
            data = load_experiment_data(exp_dir)
            if data:
                large_batch_exps.append(data)
        else:
            # Small batch experiment (cartpole without lb)
            data = load_experiment_data(exp_dir)
            if data:
                small_batch_exps.append(data)
    
    print(f"Found {len(small_batch_exps)} small batch experiments")
    print(f"Found {len(large_batch_exps)} large batch experiments")
    
    # Create output directory (same as exp_base_dir parent)
    output_dir = os.path.dirname(exp_base_dir)
    output_dir = os.path.abspath(output_dir)
    print(f"Saving plots to: {output_dir}")
    
    # Plot 1: Small batch experiments
    if small_batch_exps:
        plot_learning_curves(
            small_batch_exps,
            'Learning Curves: Small Batch Experiments (cartpole)',
            os.path.join(output_dir, 'cartpole_small_batch_learning_curves.png')
        )
    else:
        print("No small batch experiments found!")
    
    # Plot 2: Large batch experiments
    if large_batch_exps:
        plot_learning_curves(
            large_batch_exps,
            'Learning Curves: Large Batch Experiments (cartpole_lb)',
            os.path.join(output_dir, 'cartpole_large_batch_learning_curves.png')
        )
    else:
        print("No large batch experiments found!")

if __name__ == "__main__":
    main()


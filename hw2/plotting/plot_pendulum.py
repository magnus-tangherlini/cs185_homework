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

def load_pendulum_data(exp_dir):
    """Load log data from an InvertedPendulum experiment directory."""
    log_file = os.path.join(exp_dir, 'log.csv')
    if not os.path.exists(log_file):
        return None
    
    try:
        envsteps = []
        eval_returns = []
        
        with open(log_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    envstep = float(row['Train_EnvstepsSoFar'])
                    eval_return = float(row['Eval_AverageReturn'])
                    envsteps.append(envstep)
                    eval_returns.append(eval_return)
                except (ValueError, KeyError) as e:
                    continue
        
        if len(envsteps) == 0:
            return None
        
        envsteps = np.array(envsteps)
        eval_returns = np.array(eval_returns)
        
        # Filter to only include data up to 100,000 environment steps
        mask = envsteps <= 100000
        envsteps = envsteps[mask]
        eval_returns = eval_returns[mask]
        
        if len(envsteps) == 0:
            return None
        
        return {
            'envsteps': envsteps,
            'eval_returns': eval_returns,
        }
    except Exception as e:
        print(f"Error loading {log_file}: {e}")
        return None

def plot_pendulum_curves(experiments_data, output_file):
    """Plot learning curves for multiple InvertedPendulum experiments."""
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot create plots: matplotlib not available")
        return
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(experiments_data)))
        
        for i, (exp_name, exp_data) in enumerate(experiments_data.items()):
            ax.plot(
                exp_data['envsteps'],
                exp_data['eval_returns'],
                color=colors[i],
                linewidth=2,
                alpha=0.8,
                label=exp_name
            )
        
        ax.set_xlabel('Number of Environment Steps', fontsize=12)
        ax.set_ylabel('Average Return', fontsize=12)
        ax.set_title('InvertedPendulum Learning Curves (up to 100,000 steps)', fontsize=14, fontweight='bold')
        ax.set_xlim(left=0, right=100000)  # Set x-axis limit to 100,000
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
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
    
    # Find all InvertedPendulum experiments
    pendulum_exps = glob.glob(os.path.join(exp_base_dir, '*InvertedPendulum*'))
    
    if len(pendulum_exps) == 0:
        print("No InvertedPendulum experiments found!")
        return
    
    print(f"Found {len(pendulum_exps)} InvertedPendulum experiments")
    
    # Load data for all experiments
    experiments_data = {}
    for exp_dir in sorted(pendulum_exps):
        exp_name = os.path.basename(exp_dir)
        # Clean up the name for the legend
        clean_name = exp_name.replace('InvertedPendulum-v4_', '').replace('_sd1_', '_').split('_2026')[0]
        
        data = load_pendulum_data(exp_dir)
        if data is not None:
            experiments_data[clean_name] = data
            print(f"  Loaded: {clean_name}")
        else:
            print(f"  Failed to load: {clean_name}")
    
    if len(experiments_data) == 0:
        print("No valid InvertedPendulum experiment data found!")
        return
    
    # Create output directory (same as exp_base_dir parent)
    output_dir = os.path.dirname(exp_base_dir)
    output_dir = os.path.abspath(output_dir)
    print(f"\nSaving plot to: {output_dir}")
    
    # Plot learning curves
    plot_pendulum_curves(
        experiments_data,
        os.path.join(output_dir, 'inverted_pendulum_learning_curves.png')
    )

if __name__ == "__main__":
    main()


import os
import sys
import glob
import csv
import re

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

def extract_lambda_from_name(exp_name):
    """Extract lambda value from experiment name like 'lunar_lander_lambda0.95' or 'lunar_lander_lambda95'."""
    # Try to find lambda value in the name
    # Handle both formats: lambda0.95 or lambda95 (which should be 0.95)
    match = re.search(r'lambda(\d+)', exp_name, re.IGNORECASE)
    if match:
        lambda_str = match.group(1)
        # If it's a 2-digit number (like 95, 98, 99), treat it as 0.95, 0.98, 0.99
        if len(lambda_str) == 2 and lambda_str != '00':
            return float(lambda_str) / 100.0
        # Otherwise treat as integer (0, 1, etc.)
        else:
            return float(lambda_str)
    # Try decimal format like lambda0.95
    match = re.search(r'lambda([\d.]+)', exp_name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None

def load_lunar_lander_data(exp_dir):
    """Load log data from a LunarLander experiment directory."""
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
        
        return {
            'envsteps': np.array(envsteps),
            'eval_returns': np.array(eval_returns),
        }
    except Exception as e:
        print(f"Error loading {log_file}: {e}")
        return None

def plot_lunar_lander_curves(experiments_data, output_file):
    """Plot learning curves for multiple LunarLander experiments with lambda in legend."""
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot create plots: matplotlib not available")
        return
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by lambda value for consistent ordering
        sorted_experiments = sorted(experiments_data.items(), key=lambda x: x[0] if x[0] is not None else float('inf'))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_experiments)))
        
        for i, (lambda_val, exp_data) in enumerate(sorted_experiments):
            label = f'λ = {lambda_val}' if lambda_val is not None else 'No lambda'
            ax.plot(
                exp_data['envsteps'],
                exp_data['eval_returns'],
                color=colors[i],
                linewidth=2,
                alpha=0.8,
                label=label
            )
        
        ax.set_xlabel('Number of Environment Steps', fontsize=12)
        ax.set_ylabel('Average Return', fontsize=12)
        ax.set_title('LunarLander Learning Curves (by GAE Lambda)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10, title='GAE Lambda (λ)')
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
    
    # Find all LunarLander experiments
    lunar_exps = glob.glob(os.path.join(exp_base_dir, '*LunarLander*'))
    
    if len(lunar_exps) == 0:
        print("No LunarLander experiments found!")
        return
    
    print(f"Found {len(lunar_exps)} LunarLander experiments")
    
    # Load data and extract lambda values
    experiments_data = {}
    for exp_dir in sorted(lunar_exps):
        exp_name = os.path.basename(exp_dir)
        print(f"  Processing: {exp_name}")
        
        # Extract lambda value from experiment name
        lambda_val = extract_lambda_from_name(exp_name)
        
        # Load the data
        data = load_lunar_lander_data(exp_dir)
        if data is not None:
            if lambda_val is not None:
                # If multiple experiments have the same lambda, we'll use the most recent one
                # or could average them - for now, just use the most recent
                if lambda_val not in experiments_data:
                    experiments_data[lambda_val] = data
                    print(f"    Loaded: λ = {lambda_val}")
                else:
                    print(f"    Skipping duplicate lambda {lambda_val} (keeping first)")
            else:
                # If no lambda found, use None as key
                if None not in experiments_data:
                    experiments_data[None] = data
                    print(f"    Loaded: (no lambda specified)")
        else:
            print(f"    Failed to load data")
    
    if len(experiments_data) == 0:
        print("No valid LunarLander experiment data found!")
        return
    
    # Create output directory (same as exp_base_dir parent)
    output_dir = os.path.dirname(exp_base_dir)
    output_dir = os.path.abspath(output_dir)
    print(f"\nSaving plot to: {output_dir}")
    
    # Plot learning curves
    plot_lunar_lander_curves(
        experiments_data,
        os.path.join(output_dir, 'lunar_lander_lambda_comparison.png')
    )

if __name__ == "__main__":
    main()


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

def load_cheetah_baseline_data(exp_dir):
    """Load log data from a cheetah baseline experiment directory."""
    log_file = os.path.join(exp_dir, 'log.csv')
    if not os.path.exists(log_file):
        return None
    
    try:
        envsteps = []
        baseline_losses = []
        eval_returns = []
        
        with open(log_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    envstep = float(row['Train_EnvstepsSoFar'])
                    envsteps.append(envstep)
                    
                    # Baseline Loss (should exist when using baseline)
                    if 'Baseline Loss' in row and row['Baseline Loss']:
                        try:
                            baseline_loss = float(row['Baseline Loss'])
                            baseline_losses.append(baseline_loss)
                        except ValueError:
                            baseline_losses.append(np.nan)
                    else:
                        baseline_losses.append(np.nan)
                    
                    # Eval Average Return
                    eval_return = float(row['Eval_AverageReturn'])
                    eval_returns.append(eval_return)
                except (ValueError, KeyError) as e:
                    continue
        
        if len(envsteps) == 0:
            return None
        
        return {
            'envsteps': np.array(envsteps),
            'baseline_losses': np.array(baseline_losses),
            'eval_returns': np.array(eval_returns),
        }
    except Exception as e:
        print(f"Error loading {log_file}: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_baseline_loss(experiments_data, output_file):
    """Plot baseline loss learning curve for multiple experiments."""
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot create plots: matplotlib not available")
        return
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (exp_name, exp_data) in enumerate(experiments_data.items()):
            # Filter out NaN values for baseline losses
            valid_mask = ~np.isnan(exp_data['baseline_losses'])
            envsteps_valid = exp_data['envsteps'][valid_mask]
            losses_valid = exp_data['baseline_losses'][valid_mask]
            
            if len(envsteps_valid) > 0:
                ax.plot(
                    envsteps_valid,
                    losses_valid,
                    color=colors[i % len(colors)],
                    linewidth=2,
                    alpha=0.8,
                    label=exp_name
                )
        
        ax.set_xlabel('Number of Environment Steps', fontsize=12)
        ax.set_ylabel('Baseline Loss', fontsize=12)
        ax.set_title('Baseline Loss Learning Curves (Cheetah with Baseline)', fontsize=14, fontweight='bold')
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

def load_cheetah_eval_data(exp_dir):
    """Load eval return data from any cheetah experiment."""
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

def plot_eval_return(experiments_data, output_file):
    """Plot eval average return learning curve for multiple experiments."""
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
        ax.set_title('Eval Average Return Learning Curves (All Cheetah Experiments)', fontsize=14, fontweight='bold')
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
    
    # Find all cheetah experiments
    all_cheetah_exps = glob.glob(os.path.join(exp_base_dir, '*cheetah*'))
    
    if len(all_cheetah_exps) == 0:
        print("No cheetah experiments found!")
        return
    
    print(f"Found {len(all_cheetah_exps)} cheetah experiments")
    
    # Find baseline experiments (cheetah_baseline and cheetah_small_baseline)
    baseline_exps = []
    for exp_dir in all_cheetah_exps:
        exp_name = os.path.basename(exp_dir)
        if 'baseline' in exp_name.lower():
            baseline_exps.append(exp_dir)
    
    print(f"Found {len(baseline_exps)} baseline experiments")
    
    # Load baseline loss data for baseline experiments
    baseline_data = {}
    for exp_dir in sorted(baseline_exps):
        exp_name = os.path.basename(exp_dir)
        # Clean up the name for the legend
        clean_name = exp_name.replace('HalfCheetah-v4_', '').replace('_sd1_', '_').split('_2026')[0]
        data = load_cheetah_baseline_data(exp_dir)
        if data is not None:
            baseline_data[clean_name] = data
            print(f"  Loaded: {clean_name}")
    
    # Load eval return data for ALL cheetah experiments
    eval_data = {}
    for exp_dir in sorted(all_cheetah_exps):
        exp_name = os.path.basename(exp_dir)
        # Clean up the name for the legend
        clean_name = exp_name.replace('HalfCheetah-v4_', '').replace('_sd1_', '_').split('_2026')[0]
        data = load_cheetah_eval_data(exp_dir)
        if data is not None:
            eval_data[clean_name] = data
            print(f"  Loaded eval data: {clean_name}")
    
    # Create output directory (same as exp_base_dir parent)
    output_dir = os.path.dirname(exp_base_dir)
    output_dir = os.path.abspath(output_dir)
    print(f"\nSaving plots to: {output_dir}")
    
    # Plot 1: Baseline Loss (for baseline experiments only)
    if baseline_data:
        plot_baseline_loss(
            baseline_data,
            os.path.join(output_dir, 'cheetah_baseline_loss.png')
        )
    else:
        print("No baseline experiments found for baseline loss plot!")
    
    # Plot 2: Eval Average Return (for ALL cheetah experiments)
    if eval_data:
        plot_eval_return(
            eval_data,
            os.path.join(output_dir, 'cheetah_all_eval_return.png')
        )
    else:
        print("No cheetah experiments found for eval return plot!")

if __name__ == "__main__":
    main()


import numpy as np
import pandas as pd
import time
from datetime import datetime
import matplotlib.pyplot as plt

# ==============================================
# CORRECT WAY TO IMPORT CEC 2022 FUNCTIONS
# ==============================================
# Import ALL 12 CEC 2022 benchmark functions individually
from opfunu.cec_based.cec2022 import (
    F12022, F22022, F32022, F42022, F52022,
    F62022, F72022, F82022, F92022, F102022,
    F112022, F122022
)

# ==============================================
# ALGORITHM CLASS (NO CHANGES)
# ==============================================
class AdaptiveDifferentialCrossover:
    def __init__(self, max_evaluations=200000, dim=10):
        self.max_evaluations = max_evaluations
        self.dim = dim
        self.pop_size = 100
        self.F = 0.8
        self.CR = 0.7
    
    def optimize(self, func):
        # Population initialization
        pop = np.random.uniform(-100, 100, (self.pop_size, self.dim))
        fitness = np.array([func.evaluate(ind) for ind in pop])
        
        evals = self.pop_size
        best_idx = np.argmin(fitness)
        best_fit = fitness[best_idx]
        best_sol = pop[best_idx].copy()
        
        while evals < self.max_evaluations:
            for i in range(self.pop_size):
                # Parents selection
                idxs = [j for j in range(self.pop_size) if j != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                
                # Mutation and crossover
                mutant = a + self.F * (b - c)
                trial = pop[i].copy()
                cross = np.random.rand(self.dim) < self.CR
                if not np.any(cross):
                    cross[np.random.randint(self.dim)] = True
                trial[cross] = mutant[cross]
                
                # Bounds check
                trial = np.clip(trial, -100, 100)
                
                # Evaluation and selection
                trial_fit = func.evaluate(trial)
                evals += 1
                
                if trial_fit < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fit
                    
                    if trial_fit < best_fit:
                        best_fit = trial_fit
                        best_sol = trial.copy()
            
            if evals >= self.max_evaluations:
                break
        
        return best_fit, best_sol

# ==============================================
# HELPER FUNCTION: Get all CEC 2022 functions
# ==============================================
def get_all_cec2022_functions():
    """Returns list of all 12 CEC 2022 function classes"""
    return [
        F12022, F22022, F32022, F42022, F52022,
        F62022, F72022, F82022, F92022, F102022,
        F112022, F122022
    ]

# ==============================================
# MAIN TESTING FUNCTION WITH STATISTICS
# ==============================================
def comprehensive_statistical_test():
    """
    Complete statistical analysis across dimensions
    Saves results to CSV files
    """
    # Test configurations
    dimensions_to_test = [10, 20]
    num_runs_per_config = 30  # For statistical significance
    max_evaluations_per_dim = {
        10: 200000,   # 10D: 10,000 evaluations
        20: 1000000,   # 20D: 20,000 evaluations
        #30: 30000    # 30D: 30,000 evaluations
    }
    
    # Get all functions
    all_functions = get_all_cec2022_functions()
    
    # Prepare data storage
    all_results = []
    detailed_run_data = []
    
    print("=" * 90)
    print("COMPREHENSIVE STATISTICAL ANALYSIS - CEC 2022 BENCHMARK")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dimensions: {dimensions_to_test}")
    print(f"Runs per configuration: {num_runs_per_config}")
    print("=" * 90)
    
    # Main testing loop
    total_tests = len(dimensions_to_test) * len(all_functions)
    test_counter = 0
    
    for dim_idx, dimension in enumerate(dimensions_to_test):
        print(f"\n{'#' * 80}")
        print(f"DIMENSION: {dimension}D")
        print(f"{'#' * 80}")
        
        max_evaluations = max_evaluations_per_dim[dimension]
        
        for func_idx, FuncClass in enumerate(all_functions):
            test_counter += 1
            func = FuncClass(ndim=dimension)
            func_name = func.name
            func_id = func_idx + 1
            
            print(f"\n[{test_counter}/{total_tests}] Testing: {func_name} ({dimension}D)")
            print(f"  Global optimum: {func.f_global}")
            print(f"  Budget: {max_evaluations} evaluations")
            print("  " + "-" * 60)
            
            # Arrays to store results for this configuration
            run_errors = []
            run_fitness = []
            run_times = []
            
            # Multiple independent runs
            start_time = time.time()
            
            for run in range(num_runs_per_config):
                # Create algorithm instance
                adc = AdaptiveDifferentialCrossover(max_evaluations=max_evaluations, dim=dimension)
                
                # Run optimization
                run_start = time.time()
                best_fitness, _ = adc.optimize(func)
                run_elapsed = time.time() - run_start
                
                # Calculate error
                error = best_fitness - func.f_global
                
                # Store run data
                run_errors.append(error)
                run_fitness.append(best_fitness)
                run_times.append(run_elapsed)
                
                # Store detailed run data for CSV
                detailed_run_data.append({
                    'Function_ID': func_id,
                    'Function_Name': func_name,
                    'Dimension': dimension,
                    'Run_Number': run + 1,
                    'Budget': max_evaluations,
                    'Best_Fitness': best_fitness,
                    'Error': error,
                    'Run_Time': run_elapsed,
                    'Success': error < 1e-8,
                    'Optimum_Value': func.f_global
                })
                
                # Print progress every 5 runs
                if (run + 1) % 5 == 0:
                    print(f"    Run {run + 1:2d}: Error = {error:.4e}, Time = {run_elapsed:.2f}s")
            
            total_time = time.time() - start_time
            
            # Convert to numpy arrays for statistics
            errors_array = np.array(run_errors)
            fitness_array = np.array(run_fitness)
            times_array = np.array(run_times)
            
            # Calculate statistical measures
            stats_summary = {
                'Function_ID': func_id,
                'Function_Name': func_name,
                'Dimension': dimension,
                'Budget': max_evaluations,
                'Optimum': func.f_global,
                
                # Error statistics
                'Mean_Error': np.mean(errors_array),
                'Std_Error': np.std(errors_array),
                'Min_Error': np.min(errors_array),
                'Max_Error': np.max(errors_array),
                'Median_Error': np.median(errors_array),
                'IQR_Error': np.percentile(errors_array, 75) - np.percentile(errors_array, 25),
                
                # Fitness statistics
                'Mean_Fitness': np.mean(fitness_array),
                'Std_Fitness': np.std(fitness_array),
                'Best_Fitness': np.min(fitness_array),
                'Worst_Fitness': np.max(fitness_array),
                
                # Time statistics
                'Mean_Time': np.mean(times_array),
                'Std_Time': np.std(times_array),
                'Total_Time': total_time,
                
                # Success metrics
                'Success_Rate': np.sum(errors_array < 1e-8) / num_runs_per_config * 100,
                'Success_Count': np.sum(errors_array < 1e-8),
                
                # Additional statistical measures
                'RMSE': np.sqrt(np.mean(errors_array**2)),
                'MAE': np.mean(np.abs(errors_array)),
                'Stability_Index': np.std(errors_array) / np.mean(np.abs(errors_array)) if np.mean(np.abs(errors_array)) > 0 else 0
            }
            
            all_results.append(stats_summary)
            
            # Print configuration summary
            print(f"\n  [SUMMARY] {func_name} ({dimension}D):")
            print(f"    Success Rate: {stats_summary['Success_Rate']:.1f}% ({stats_summary['Success_Count']}/{num_runs_per_config})")
            print(f"    Mean Error:   {stats_summary['Mean_Error']:.4e} ± {stats_summary['Std_Error']:.4e}")
            print(f"    Min/Max:      {stats_summary['Min_Error']:.4e} / {stats_summary['Max_Error']:.4e}")
            print(f"    Time/Run:     {stats_summary['Mean_Time']:.2f}s")
    
    # Create DataFrames
    summary_df = pd.DataFrame(all_results)
    detailed_df = pd.DataFrame(detailed_run_data)
    
    # Save to CSV files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary_filename = f"cec2022_statistical_summary_{timestamp}.csv"
    detailed_filename = f"cec2022_detailed_runs_{timestamp}.csv"
    
    summary_df.to_csv(summary_filename, index=False, float_format='%.6e')
    detailed_df.to_csv(detailed_filename, index=False, float_format='%.6e')
    
    print(f"\n{'=' * 90}")
    print("ANALYSIS COMPLETE!")
    print(f"{'=' * 90}")
    print(f"Summary statistics saved to: {summary_filename}")
    print(f"Detailed run data saved to: {detailed_filename}")
    
    # Generate visualizations
    generate_visualizations(summary_df, timestamp)
    
    return summary_df, detailed_df

# ==============================================
# VISUALIZATION FUNCTIONS
# ==============================================
def generate_visualizations(summary_df, timestamp):
    """Generate visualization plots"""
    print("\nGenerating visualizations...")
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: Success rate by dimension
    ax1 = plt.subplot(2, 3, 1)
    success_by_dim = summary_df.groupby('Dimension')['Success_Rate'].mean()
    success_by_dim.plot(kind='bar', color='skyblue', ax=ax1)
    ax1.set_title('Success Rate by Dimension', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Dimension')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_ylim([0, 100])
    
    # Add value labels
    for i, v in enumerate(success_by_dim):
        ax1.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=9)
    
    # Plot 2: Mean error by function
    ax2 = plt.subplot(2, 3, 2)
    mean_errors = summary_df.groupby('Function_ID')['Mean_Error'].mean()
    ax2.plot(mean_errors.index, mean_errors.values, 'b-o', linewidth=2, markersize=6)
    ax2.set_title('Mean Error by Function', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Function ID')
    ax2.set_ylabel('Mean Error (log scale)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error distribution boxplot
    ax3 = plt.subplot(2, 3, 3)
    error_data = []
    labels = []
    for dim in sorted(summary_df['Dimension'].unique()):
        dim_data = summary_df[summary_df['Dimension'] == dim]['Mean_Error']
        error_data.append(dim_data)
        labels.append(f'{dim}D')
    
    bp = ax3.boxplot(error_data, labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax3.set_title('Error Distribution by Dimension', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Mean Error (log scale)')
    ax3.set_yscale('log')
    
    # Plot 4: Runtime analysis
    ax4 = plt.subplot(2, 3, 4)
    runtime_by_dim = summary_df.groupby('Dimension')['Mean_Time'].mean()
    runtime_by_dim.plot(kind='bar', color='orange', ax=ax4)
    ax4.set_title('Average Runtime by Dimension', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Dimension')
    ax4.set_ylabel('Time (seconds)')
    
    # Plot 5: Success rate heatmap
    ax5 = plt.subplot(2, 3, 5)
    pivot_data = summary_df.pivot_table(
        values='Success_Rate', 
        index='Function_ID', 
        columns='Dimension'
    )
    im = ax5.imshow(pivot_data.values, cmap='YlGnBu', aspect='auto')
    ax5.set_title('Success Rate Heatmap', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Dimension')
    ax5.set_ylabel('Function ID')
    plt.colorbar(im, ax=ax5, label='Success Rate (%)')
    
    # Plot 6: Convergence rate vs dimension
    ax6 = plt.subplot(2, 3, 6)
    convergence_data = []
    for dim in sorted(summary_df['Dimension'].unique()):
        dim_df = summary_df[summary_df['Dimension'] == dim]
        convergence_rate = dim_df['Success_Rate'].mean()
        convergence_data.append(convergence_rate)
    
    ax6.plot(sorted(summary_df['Dimension'].unique()), convergence_data, 
             'r-s', linewidth=2, markersize=8)
    ax6.set_title('Convergence Rate vs Dimension', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Dimension')
    ax6.set_ylabel('Convergence Rate (%)')
    ax6.set_ylim([0, 100])
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'cec2022_analysis_plots_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualizations saved to: cec2022_analysis_plots_{timestamp}.png")

# ==============================================
# QUICK TEST FUNCTION (SINGLE RUN)
# ==============================================
def quick_test_single_function():
    """Quick test for single function"""
    # Test on F1 function, 10 dimensions
    func = F12022(ndim=10)
    adc = AdaptiveDifferentialCrossover(budget=1000)
    
    start_time = time.time()
    best_f, best_x = adc.optimize(func)
    elapsed_time = time.time() - start_time
    
    print(f"Function: {func.name}")
    print(f"Optimum: {func.f_global}")
    print(f"Found: {best_f:.10f}")
    print(f"Error: {best_f - func.f_global:.6e}")
    print(f"Time: {elapsed_time:.2f} seconds")
    
    return best_f, best_x

# ==============================================
# MAIN EXECUTION
# ==============================================
if __name__ == "__main__":
    print("Adaptive Differential Crossover - CEC 2022 Testing")
    print("=" * 70)
    
    # Uncomment ONE of these options:
    
    # Option 1: Quick single test (fast, for debugging)
    # print("\nRunning quick single test...")
    # best_f, best_x = quick_test_single_function()
    
    # Option 2: Comprehensive statistical analysis (RECOMMENDED)
    print("\nRunning comprehensive statistical analysis...")
    print("This will take time (25 runs × 12 functions × 3 dimensions)")
    print("Estimated time: 15-30 minutes depending on your system")
    
    # Ask for confirmation
    response = input("Continue? (y/n): ")
    if response.lower() == 'y':
        summary_df, detailed_df = comprehensive_statistical_test()
        
        # Display final summary
        print(f"\n{'=' * 90}")
        print("FINAL SUMMARY")
        print(f"{'=' * 90}")
        
        total_runs = len(detailed_df)
        total_success = detailed_df['Success'].sum()
        overall_success_rate = total_success / total_runs * 100
        
        print(f"Total runs performed: {total_runs:,}")
        print(f"Total successful runs: {total_success:,}")
        print(f"Overall success rate: {overall_success_rate:.2f}%")
        print(f"Average error across all runs: {detailed_df['Error'].abs().mean():.4e}")
        print(f"Total computation time: {detailed_df['Run_Time'].sum():.2f} seconds")
        
        # Show dimension-wise summary
        print(f"\nDimension-wise Performance:")
        dim_summary = detailed_df.groupby('Dimension').agg({
            'Success': 'mean',
            'Error': lambda x: x.abs().mean(),
            'Run_Time': 'mean'
        }).round(4)
        print(dim_summary)
    else:
        print("Test cancelled.")
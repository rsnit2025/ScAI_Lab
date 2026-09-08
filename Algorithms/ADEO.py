import numpy as np
import pandas as pd
import time
from datetime import datetime
import matplotlib.pyplot as plt

# ==============================================
# CORRECT WAY TO IMPORT CEC 2022 FUNCTIONS
# ==============================================
from opfunu.cec_based.cec2022 import (
    F12022, F22022, F32022, F42022, F52022,
    F62022, F72022, F82022, F92022, F102022,
    F112022, F122022
)

# ==============================================
# UPDATED ALGORITHM CLASS (YOUR NEW ALGORITHM)
# ==============================================
class AdaptiveDifferentialEvolutionOptimizer:
    def __init__(self, max_evaluations=200000, dim=10, pop_size=100, F=0.8, CR=0.9, adapt_factor=0.99):
        self.max_evaluations = max_evaluations
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.adapt_factor = adapt_factor  # Factor for adapting F and CR
        self.bounds = (-100.0, 100.0)
        self.eval_count = 0
        self.f_opt = None
        self.x_opt = None
    
    def optimize(self, func):
        """Optimization method compatible with testing framework"""
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func.evaluate(ind) for ind in population])
        self.eval_count = self.pop_size
        
        # Reset F and CR to initial values for each run
        current_F = self.F
        current_CR = self.CR
        
        # Evolutionary loop
        while self.eval_count < self.max_evaluations:
            for i in range(self.pop_size):
                # Mutation
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + current_F * (b - c), self.bounds[0], self.bounds[1])
                
                # Crossover
                cross_points = np.random.rand(self.dim) < current_CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Selection
                f_trial = func.evaluate(trial)
                self.eval_count += 1
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                
                if self.eval_count >= self.max_evaluations:
                    break
            
            # Adapt F and CR
            current_F *= self.adapt_factor
            current_CR *= self.adapt_factor
        
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]
        
        return self.f_opt, self.x_opt

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
    dimensions_to_test = [10, 20] #  30 
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
    print("ADAPTIVE DIFFERENTIAL EVOLUTION OPTIMIZER - CEC 2022 BENCHMARK")
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
            print(f"  Max_evaluations: {max_evaluations} evaluations")
            print("  " + "-" * 60)
            
            # Arrays to store results for this configuration
            run_errors = []
            run_fitness = []
            run_times = []
            final_F_values = []
            final_CR_values = []
            
            # Multiple independent runs
            start_time = time.time()
            
            for run in range(num_runs_per_config):
                # Create algorithm instance with adaptive parameters
                ade = AdaptiveDifferentialEvolutionOptimizer(
                    max_evaluations=max_evaluations, 
                    dim=dimension,
                    pop_size=100,
                    F=0.8,
                    CR=0.9,
                    adapt_factor=0.99
                )
                
                # Run optimization
                run_start = time.time()
                best_fitness, _ = ade.optimize(func)
                run_elapsed = time.time() - run_start
                
                # Calculate error
                error = best_fitness - func.f_global
                
                # Store run data
                run_errors.append(error)
                run_fitness.append(best_fitness)
                run_times.append(run_elapsed)
                
                # Calculate final adapted parameter values (approximate)
                iterations = (max_evaluations - 50) // 50  # Approximate number of generations
                final_F = 0.8 * (0.99 ** iterations)
                final_CR = 0.9 * (0.99 ** iterations)
                final_F_values.append(final_F)
                final_CR_values.append(final_CR)
                
                # Store detailed run data for CSV
                detailed_run_data.append({
                    'Function_ID': func_id,
                    'Function_Name': func_name,
                    'Dimension': dimension,
                    'Run_Number': run + 1,
                    'Max_evaluations': max_evaluations,
                    'Best_Fitness': best_fitness,
                    'Error': error,
                    'Run_Time': run_elapsed,
                    'Success': error < 1e-8,
                    'Optimum_Value': func.f_global,
                    'Final_F': final_F,
                    'Final_CR': final_CR,
                    'Evaluations_Used': ade.eval_count
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
                'Max_evaluations': max_evaluations,
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
                
                # Parameter adaptation statistics
                'Mean_Final_F': np.mean(final_F_values),
                'Mean_Final_CR': np.mean(final_CR_values),
                
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
            print(f"    Final F/CR:   {stats_summary['Mean_Final_F']:.3f} / {stats_summary['Mean_Final_CR']:.3f}")
            print(f"    Time/Run:     {stats_summary['Mean_Time']:.2f}s")
    
    # Create DataFrames
    summary_df = pd.DataFrame(all_results)
    detailed_df = pd.DataFrame(detailed_run_data)
    
    # Save to CSV files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary_filename = f"adaptive_de_statistical_summary_{timestamp}.csv"
    detailed_filename = f"adaptive_de_detailed_runs_{timestamp}.csv"
    
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
    
    # Plot 3: Parameter adaptation (Final F vs Final CR)
    ax3 = plt.subplot(2, 3, 3)
    dim_colors = {10: 'blue', 20: 'green', 30: 'red'}
    for dim in summary_df['Dimension'].unique():
        dim_data = summary_df[summary_df['Dimension'] == dim]
        ax3.scatter(dim_data['Mean_Final_F'], dim_data['Mean_Final_CR'], 
                   color=dim_colors[dim], label=f'{dim}D', s=80, alpha=0.7)
    
    ax3.set_title('Parameter Adaptation (Final Values)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Final F Value')
    ax3.set_ylabel('Final CR Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
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
    
    # Plot 6: Error vs Dimension trend
    ax6 = plt.subplot(2, 3, 6)
    for func_id in summary_df['Function_ID'].unique()[:5]:  # First 5 functions
        func_data = summary_df[summary_df['Function_ID'] == func_id]
        ax6.plot(func_data['Dimension'], func_data['Mean_Error'], 
                marker='o', label=f'F{func_id}')
    
    ax6.set_title('Error vs Dimension (Sample Functions)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Dimension')
    ax6.set_ylabel('Mean Error (log scale)')
    ax6.set_yscale('log')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'adaptive_de_analysis_plots_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualizations saved to: adaptive_de_analysis_plots_{timestamp}.png")

# ==============================================
# QUICK TEST FUNCTION (SINGLE RUN)
# ==============================================
def quick_test_single_function():
    """Quick test for single function"""
    # Test on F1 function, 10 dimensions
    func = F12022(ndim=10)
    
    # Create algorithm with adaptive parameters
    ade = AdaptiveDifferentialEvolutionOptimizer(
        max_evaluations=200000, 
        dim=10,
        pop_size=100,
        F=0.8,
        CR=0.9,
        adapt_factor=0.99
    )
    
    start_time = time.time()
    best_f, best_x = ade.optimize(func)
    elapsed_time = time.time() - start_time
    
    print(f"Algorithm: Adaptive Differential Evolution Optimizer")
    print(f"Function: {func.name}")
    print(f"Optimum: {func.f_global}")
    print(f"Found: {best_f:.10f}")
    print(f"Error: {best_f - func.f_global:.6e}")
    print(f"Evaluations used: {ade.eval_count}")
    print(f"Time: {elapsed_time:.2f} seconds")
    
    return best_f, best_x

# ==============================================
# PARAMETER SENSITIVITY ANALYSIS
# ==============================================
def parameter_sensitivity_analysis():
    """Test different parameter combinations"""
    print("\nParameter Sensitivity Analysis")
    print("=" * 60)
    
    func = F12022(ndim=10)
    max_evaluations = 200000
    
    # Different parameter combinations
    param_combinations = [
        {'F': 0.5, 'CR': 0.7, 'adapt_factor': 0.95},
        {'F': 0.8, 'CR': 0.9, 'adapt_factor': 0.99},
        {'F': 1.0, 'CR': 0.5, 'adapt_factor': 0.90},
        {'F': 0.6, 'CR': 0.8, 'adapt_factor': 0.97},
    ]
    
    results = []
    
    for i, params in enumerate(param_combinations):
        print(f"\nTest {i+1}: F={params['F']}, CR={params['CR']}, adapt={params['adapt_factor']}")
        
        run_errors = []
        for run in range(5):  # 5 runs per configuration
            ade = AdaptiveDifferentialEvolutionOptimizer(
                max_evaluations=max_evaluations,
                dim=10,
                pop_size=100,
                F=params['F'],
                CR=params['CR'],
                adapt_factor=params['adapt_factor']
            )
            
            best_fitness, _ = ade.optimize(func)
            error = best_fitness - func.f_global
            run_errors.append(error)
        
        mean_error = np.mean(run_errors)
        std_error = np.std(run_errors)
        
        results.append({
            **params,
            'Mean_Error': mean_error,
            'Std_Error': std_error
        })
        
        print(f"  Mean Error: {mean_error:.4e} ± {std_error:.4e}")
    
    # Display results
    results_df = pd.DataFrame(results)
    print(f"\n{'=' * 60}")
    print("Parameter Sensitivity Results:")
    print(results_df.to_string(index=False))
    
    return results_df

# ==============================================
# MAIN EXECUTION
# ==============================================
if __name__ == "__main__":
    print("Adaptive Differential Evolution Optimizer - CEC 2022 Testing")
    print("=" * 70)
    
    # Uncomment ONE of these options:
    
    # Option 1: Quick single test (fast, for debugging)
    # print("\nRunning quick single test...")
    # best_f, best_x = quick_test_single_function()
    
    # Option 2: Parameter sensitivity analysis
    # print("\nRunning parameter sensitivity analysis...")
    # param_results = parameter_sensitivity_analysis()
    
    # Option 3: Comprehensive statistical analysis (RECOMMENDED)
    print("\nRunning comprehensive statistical analysis...")
    print("This will test Adaptive Differential Evolution Optimizer")
    print("30 runs × 12 functions × 2 dimensions = 720 total runs")
    print("Estimated time: 15-30 minutes depending on your system")
    
    # Ask for confirmation
    response = input("Continue? (y/n): ")
    if response.lower() == 'y':
        summary_df, detailed_df = comprehensive_statistical_test()
        
        # Display final summary
        print(f"\n{'=' * 90}")
        print("FINAL SUMMARY - ADAPTIVE DIFFERENTIAL EVOLUTION")
        print(f"{'=' * 90}")
        
        total_runs = len(detailed_df)
        total_success = detailed_df['Success'].sum()
        overall_success_rate = total_success / total_runs * 100
        
        print(f"Total runs performed: {total_runs:,}")
        print(f"Total successful runs: {total_success:,}")
        print(f"Overall success rate: {overall_success_rate:.2f}%")
        print(f"Average error across all runs: {detailed_df['Error'].abs().mean():.4e}")
        print(f"Total computation time: {detailed_df['Run_Time'].sum():.2f} seconds")
        
        # Show algorithm-specific statistics
        print(f"\nAlgorithm Parameters Summary:")
        print(f"Initial F: 0.8, Initial CR: 0.9, Adaptation Factor: 0.99")
        print(f"Average Final F: {detailed_df['Final_F'].mean():.3f}")
        print(f"Average Final CR: {detailed_df['Final_CR'].mean():.3f}")
        
        # Show dimension-wise summary
        print(f"\nDimension-wise Performance:")
        dim_summary = detailed_df.groupby('Dimension').agg({
            'Success': 'mean',
            'Error': lambda x: x.abs().mean(),
            'Run_Time': 'mean',
            'Final_F': 'mean',
            'Final_CR': 'mean'
        }).round(4)
        print(dim_summary)
    else:
        print("Test cancelled.")
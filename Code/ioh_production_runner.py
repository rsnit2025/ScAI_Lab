"""
Production-scale IOHanalyzer logging for all 10 DE algorithms in this repo.

Usage:
    python ioh_production_runner.py <algorithm_key>

<algorithm_key> is one of the keys in ALGORITHM_CONFIGS below. Each key maps
to the exact class + constructor kwargs used in that algorithm's own
comprehensive_statistical_test() function (i.e. the "actual configuration"
already defined in each script), so results are directly comparable to the
existing CSV outputs in this repo.

Production settings match every script's own comprehensive_statistical_test():
    dimensions_to_test = [10, 20]
    num_runs_per_config = 30
    max_evaluations_per_dim = {10: 200000, 20: 1000000}

Run one process per algorithm (in parallel, one per CPU core) rather than
looping over all 10 in a single process, so a crash in one algorithm doesn't
take down the others and progress can be monitored independently:

    python ioh_production_runner.py adeo &
    python ioh_production_runner.py de_crossover &
    ... (one per key) ...

Output per algorithm:
    ioh_data_all/<algorithm_key>/           IOHanalyzer-compatible logs
    production_results_<algorithm_key>.csv  Per-run results (feeds SHAP later)
"""
import importlib
import re
import sys
import time

import numpy as np
import pandas as pd
import ioh

from opfunu.cec_based.cec2022 import (
    F12022, F22022, F32022, F42022, F52022,
    F62022, F72022, F82022, F92022, F102022,
    F112022, F122022
)

CEC2022_FUNCTIONS = [
    F12022, F22022, F32022, F42022, F52022,
    F62022, F72022, F82022, F92022, F102022,
    F112022, F122022
]

BOUNDS = (-100.0, 100.0)
DIMENSIONS_TO_TEST = [10, 20]
NUM_RUNS_PER_CONFIG = 30
MAX_EVALUATIONS_PER_DIM = {10: 200000, 20: 1000000}

# module, class name, and the exact extra constructor kwargs each script uses
# in its own comprehensive_statistical_test() (excluding max_evaluations/dim,
# which are swept per DIMENSIONS_TO_TEST / MAX_EVALUATIONS_PER_DIM above).
ALGORITHM_CONFIGS = {
    "adeo": dict(
        module="ADEO", cls="AdaptiveDifferentialEvolutionOptimizer",
        kwargs=dict(pop_size=100, F=0.8, CR=0.9, adapt_factor=0.99),
    ),
    "de_crossover": dict(
        module="de_crossover", cls="AdaptiveDifferentialCrossover",
        kwargs=dict(),
    ),
    "de_gradient_boost": dict(
        module="de_gradientBoost", cls="AdaptiveDifferentialEvolutionWithGradientBoost",
        kwargs=dict(population_size=100, crossover_rate=0.7, mutation_factor=0.8),
    ),
    "de_harmony_search": dict(
        module="de_harmoniSearch", cls="AdaptiveDifferentialEvolutionHarmonySearch",
        kwargs=dict(harmony_memory_size=20, hmcr=0.9, par=0.4, bw=0.5,
                    bw_decay=0.95, f_weight=0.8, cr=0.9),
    ),
    "de_local_search": dict(
        module="de_localSearch", cls="AdaptiveDifferentialEvolutionWithLocalSearch",
        kwargs=dict(population_size=100),
    ),
    "de_memetic_search": dict(
        module="de_memeticSearch", cls="AdaptiveDifferentialEvolutionWithMemeticSearch",
        kwargs=dict(),
    ),
    "de_perturbation": dict(
        module="de_perturbation", cls="AdaptiveDifferentialEvolutionWithAdaptivePerturbation",
        kwargs=dict(population_size=100, crossover_rate=0.7, mutation_factor=0.8),
    ),
    "de_plus": dict(
        module="de_plus", cls="AdaptiveDifferentialEvolutionPlus",
        kwargs=dict(population_size=100),
    ),
    "de_pso": dict(
        module="de_pso", cls="AdaptiveDifferentialEvolutionPSO",
        kwargs=dict(),
    ),
    "dynamic_pop_v2": dict(
        module="dynamic_pop_v2", cls="AdaptiveDifferentialEvolutionWithDynamicPopulationV2",
        kwargs=dict(init_population_size=100, scaling_factor_range=(0.5, 2.0),
                    crossover_rate_range=(0.1, 1.0)),
    ),
}


class IOHFuncAdapter:
    """Makes an ioh-wrapped problem look like an opfunu function to the
    algorithm classes, which all call func.evaluate(x) rather than func(x)."""

    def __init__(self, problem):
        self._problem = problem

    def evaluate(self, x):
        return self._problem(x)


def load_algorithm_class(algo_key):
    config = ALGORITHM_CONFIGS[algo_key]
    module = importlib.import_module(config["module"])
    return getattr(module, config["cls"]), config["kwargs"]


def wrap_cec_function(func_class, dimension, algo_key):
    func = func_class(ndim=dimension)
    f_global = func.f_global
    safe_name = re.sub(r"[^A-Za-z0-9_]", "_", func.name)

    def objective(x):
        return float(func.evaluate(np.array(x)))

    def calculate_objective(instance, dim):
        return ioh.RealSolution([0.0] * dim, f_global)

    problem = ioh.wrap_problem(
        objective,
        name=f"{algo_key}_{safe_name}_{dimension}D",
        dimension=dimension,
        problem_class=ioh.ProblemClass.REAL,
        lb=BOUNDS[0],
        ub=BOUNDS[1],
        calculate_objective=calculate_objective,
    )
    return problem, func


def run_production_sweep(algo_key):
    algo_class, extra_kwargs = load_algorithm_class(algo_key)

    logger = ioh.logger.Analyzer(
        root=".",
        folder_name=f"ioh_data_all/{algo_key}",
        algorithm_name=algo_key,
        algorithm_info=f"{algo_class.__name__} (production config) on CEC2022",
    )

    records = []
    total_configs = len(DIMENSIONS_TO_TEST) * len(CEC2022_FUNCTIONS)
    config_counter = 0

    for dimension in DIMENSIONS_TO_TEST:
        max_evaluations = MAX_EVALUATIONS_PER_DIM[dimension]

        for func_idx, func_class in enumerate(CEC2022_FUNCTIONS):
            config_counter += 1
            func_id = func_idx + 1
            problem, func = wrap_cec_function(func_class, dimension, algo_key)
            problem.attach_logger(logger)
            adapter = IOHFuncAdapter(problem)

            print(f"[{algo_key}] [{config_counter}/{total_configs}] "
                  f"F{func_id} {func.name} dim={dimension} budget={max_evaluations}", flush=True)

            config_start = time.time()
            for run in range(NUM_RUNS_PER_CONFIG):
                problem.set_instance(run + 1)
                problem.reset()
                optimizer = algo_class(max_evaluations=max_evaluations, dim=dimension, **extra_kwargs)

                run_start = time.time()
                best_fitness, _ = optimizer.optimize(adapter)
                run_elapsed = time.time() - run_start

                error = best_fitness - func.f_global
                records.append({
                    "Algorithm": algo_key,
                    "Function_ID": func_id,
                    "Function_Name": func.name,
                    "Dimension": dimension,
                    "Run_Number": run + 1,
                    "Max_Evaluations": max_evaluations,
                    "Best_Fitness": best_fitness,
                    "Error": max(error, 0.0),
                    "Success": error < 1e-8,
                    "Run_Time": run_elapsed,
                })

                if (run + 1) % 10 == 0:
                    print(f"    [{algo_key}] F{func_id} dim={dimension} run {run + 1}/{NUM_RUNS_PER_CONFIG} "
                          f"error={max(error, 0.0):.4e}", flush=True)

            print(f"  [{algo_key}] F{func_id} dim={dimension} done in {time.time() - config_start:.1f}s", flush=True)

    logger.close()

    results_df = pd.DataFrame(records)
    out_csv = f"production_results_{algo_key}.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"[{algo_key}] DONE. Saved {out_csv} and ioh_data_all/{algo_key}/", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ALGORITHM_CONFIGS:
        valid_keys = ", ".join(ALGORITHM_CONFIGS.keys())
        print(f"Usage: python ioh_production_runner.py <algorithm_key>\nValid keys: {valid_keys}")
        sys.exit(1)

    run_production_sweep(sys.argv[1])

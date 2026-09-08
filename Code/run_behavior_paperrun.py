"""
Paper-scale behavior-analysis run, split one process per algorithm so all 10
can run in parallel (this machine has 12 logical cores).

Runs the (dim, max_eval) pairing that matches the original production sweep
(ioh_production_runner.py's MAX_EVALUATIONS_PER_DIM: dim10->200k, dim20->1M),
with a reduced run count since this is for descriptive behavior/trajectory
plots, not the final-fitness statistics (those already exist with 30 runs in
all_algorithms_production_combined.csv).

Usage:
    python run_behavior_paperrun.py <algorithm_key>

<algorithm_key> is one of the keys in analysis.AlgorithmTester().algorithms.
"""
import sys
import time

import pandas as pd

from analysis import AlgorithmTester

RUNS_PER_CONFIG = 5
DIM_TO_MAX_EVAL = {10: 200000, 20: 1000000}


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_behavior_paperrun.py <algorithm_key>")
        sys.exit(1)

    algo_key = sys.argv[1]
    tester = AlgorithmTester()
    if algo_key not in tester.algorithms:
        print(f"Unknown algorithm key '{algo_key}'. Valid keys: {list(tester.algorithms)}")
        sys.exit(1)

    algo_class = tester.algorithms[algo_key]
    results = []

    total_configs = len(DIM_TO_MAX_EVAL) * len(tester.functions) * RUNS_PER_CONFIG
    done = 0
    t_start = time.time()

    for dim, max_eval in DIM_TO_MAX_EVAL.items():
        for func_name, func_class in tester.functions.items():
            for run in range(RUNS_PER_CONFIG):
                t0 = time.time()
                result = tester.run_single_experiment(algo_class, func_class, dim, max_eval, run)
                done += 1
                print(f"[{algo_key}] {done}/{total_configs} dim={dim} {func_name} "
                      f"run={run+1}/{RUNS_PER_CONFIG} best={result['best_fitness']:.4g} "
                      f"({time.time()-t0:.1f}s)", flush=True)
                results.append(result)

    out_df = pd.DataFrame(results)
    out_csv = f"behavior_paperrun_{algo_key}.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[{algo_key}] DONE in {time.time()-t_start:.1f}s. Saved {out_csv}", flush=True)


if __name__ == "__main__":
    main()

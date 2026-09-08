"""
SHAP Option 1 (hyperparameter sensitivity) for every algorithm's OWN real
hyperparameters (not invented ones) — extends shap_parameter_sensitivity.py
(originally built for ADEO only) to all 10 algorithms.

For each algorithm, this samples its own actual constructor/instance
parameters (e.g. F/CR/adapt_factor for ADEO, hmcr/par/bw for harmony search,
c1/c2/w for the PSO variant, ...), runs it across random (function, dimension)
combinations at a reduced evaluation budget, trains a RandomForestRegressor
per algorithm, and explains it with SHAP.

This is intentionally run at a SMALLER budget (8000 evaluations, 2 reps)
than the production IOHanalyzer benchmark (200k/1M evaluations, 30 reps) --
a full hyperparameter *grid sweep* at production evaluation counts would
multiply the already-long production run by another 100-300x, which is a
different (and far more expensive) kind of experiment than "run the fixed
default config 30 times". See ioh_production_runner.py for the production
benchmark this pairs with.

de_memetic_search is intentionally excluded: its F/CR/population_size are
hardcoded inside optimize() with no externally settable knobs, so there is
nothing to run a sensitivity sweep over.
"""
import numpy as np
import pandas as pd

from ioh_production_runner import CEC2022_FUNCTIONS, load_algorithm_class

N_SAMPLES = 150
MAX_EVALUATIONS = 8000
RUNS_PER_SAMPLE = 2
RANDOM_SEED = 42

# (low, high) for floats, (low, high, "int") for integers sampled inclusively.
# Applied via setattr AFTER construction, so this works uniformly whether the
# parameter is a constructor kwarg or a hardcoded self.attribute internally.
ALGO_PARAM_SPACES = {
    "adeo": {
        "F": (0.3, 1.0), "CR": (0.1, 1.0),
        "adapt_factor": (0.90, 1.0), "pop_size": (20, 150, "int"),
    },
    "de_crossover": {
        "F": (0.3, 1.0), "CR": (0.1, 1.0), "pop_size": (20, 150, "int"),
    },
    "de_gradient_boost": {
        "mutation_factor": (0.3, 1.0), "crossover_rate": (0.1, 1.0),
        "population_size": (20, 150, "int"), "base_lr": (0.01, 0.3),
    },
    "de_harmony_search": {
        "hmcr": (0.5, 0.99), "par": (0.05, 0.6), "bw": (0.1, 1.0),
        "f_weight": (0.3, 1.0), "cr": (0.1, 1.0),
        "harmony_memory_size": (10, 60, "int"),
    },
    "de_local_search": {
        "population_size": (10, 100, "int"),
    },
    "de_perturbation": {
        "mutation_factor": (0.3, 1.0), "crossover_rate": (0.1, 1.0),
        "population_size": (20, 150, "int"), "base_lr": (0.01, 0.3),
    },
    "de_plus": {
        "population_size": (10, 100, "int"),
        "mutation_factor": (0.3, 1.0), "crossover_probability": (0.1, 1.0),
    },
    "de_pso": {
        "c1": (0.5, 2.5), "c2": (0.5, 2.5), "w": (0.1, 0.9),
        "initial_pop_size": (20, 120, "int"), "min_pop_size": (5, 40, "int"),
    },
    "dynamic_pop_v2": {
        "init_population_size": (20, 150, "int"),
        "scaling_factor_low": (0.1, 1.0), "scaling_factor_high": (1.0, 3.0),
        "crossover_rate_low": (0.0, 0.5), "crossover_rate_high": (0.5, 1.0),
    },
}


def apply_params(optimizer, algo_key, params):
    """setattr each sampled hyperparameter onto the constructed optimizer,
    special-casing dynamic_pop_v2's tuple-valued range parameters."""
    if algo_key == "dynamic_pop_v2":
        optimizer.scaling_factor_range = (params["scaling_factor_low"], params["scaling_factor_high"])
        optimizer.crossover_rate_range = (params["crossover_rate_low"], params["crossover_rate_high"])
        optimizer.init_population_size = int(params["init_population_size"])
        return
    for name, value in params.items():
        space = ALGO_PARAM_SPACES[algo_key][name]
        if len(space) == 3 and space[2] == "int":
            value = int(value)
        setattr(optimizer, name, value)


def sample_params(rng, algo_key, n_samples):
    space = ALGO_PARAM_SPACES[algo_key]
    data = {}
    for name, bounds in space.items():
        low, high = bounds[0], bounds[1]
        if len(bounds) == 3 and bounds[2] == "int":
            data[name] = rng.integers(low, high + 1, n_samples)
        else:
            data[name] = rng.uniform(low, high, n_samples)
    data["dimension"] = rng.choice([10, 20], n_samples)
    data["function_id"] = rng.integers(1, 13, n_samples)
    return pd.DataFrame(data)


def run_sweep_for_algorithm(algo_key):
    rng = np.random.default_rng(RANDOM_SEED)
    algo_class, _ = load_algorithm_class(algo_key)
    param_names = list(ALGO_PARAM_SPACES[algo_key].keys())
    params_df = sample_params(rng, algo_key, N_SAMPLES)

    records = []
    for row_idx, row in params_df.iterrows():
        func_class = CEC2022_FUNCTIONS[int(row["function_id"]) - 1]
        func = func_class(ndim=int(row["dimension"]))
        param_values = {name: row[name] for name in param_names}

        errors = []
        for _ in range(RUNS_PER_SAMPLE):
            optimizer = algo_class(max_evaluations=MAX_EVALUATIONS, dim=int(row["dimension"]))
            apply_params(optimizer, algo_key, param_values)
            best_fitness, _ = optimizer.optimize(func)
            errors.append(max(best_fitness - func.f_global, 0.0))

        mean_error = float(np.mean(errors))
        records.append({**row.to_dict(), "mean_error": mean_error, "log_error": np.log1p(mean_error)})

        if (row_idx + 1) % 25 == 0:
            print(f"  [{algo_key}] sweep progress: {row_idx + 1}/{N_SAMPLES}", flush=True)

    return pd.DataFrame(records), param_names


def analyze_with_shap(results_df, feature_cols, algo_key):
    from sklearn.ensemble import RandomForestRegressor
    import shap
    import matplotlib.pyplot as plt

    X = results_df[feature_cols]
    y = results_df["log_error"]

    model = RandomForestRegressor(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X, y)
    r2 = model.score(X, y)
    print(f"[{algo_key}] Model R^2 on training data: {r2:.3f}", flush=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"shap_sensitivity_{algo_key}_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(f"shap_sensitivity_{algo_key}_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()

    mean_abs_shap = pd.Series(
        np.abs(shap_values.values).mean(axis=0), index=feature_cols
    ).sort_values(ascending=False)

    return r2, mean_abs_shap


def main():
    ranking_rows = []
    for algo_key in ALGO_PARAM_SPACES:
        print(f"\n=== {algo_key}: sampling {N_SAMPLES} hyperparameter combinations ===", flush=True)
        results_df, param_names = run_sweep_for_algorithm(algo_key)
        results_df.to_csv(f"sensitivity_sweep_{algo_key}.csv", index=False)

        feature_cols = param_names + ["dimension", "function_id"]
        r2, mean_abs_shap = analyze_with_shap(results_df, feature_cols, algo_key)
        mean_abs_shap.to_csv(f"shap_sensitivity_{algo_key}_ranking.csv", header=["mean_abs_shap"])

        print(f"[{algo_key}] ranking:\n{mean_abs_shap.to_string()}", flush=True)
        for feature, value in mean_abs_shap.items():
            ranking_rows.append({"Algorithm": algo_key, "Feature": feature, "Mean_Abs_SHAP": value, "Model_R2": r2})

    combined = pd.DataFrame(ranking_rows)
    combined.to_csv("shap_sensitivity_all_algorithms_combined.csv", index=False)
    print("\nAll algorithms done. Combined ranking saved to shap_sensitivity_all_algorithms_combined.csv", flush=True)
    print("Note: de_memetic_search skipped (no externally settable hyperparameters).")


if __name__ == "__main__":
    main()

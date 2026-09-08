"""
Hyperparameter sensitivity analysis for AdaptiveDifferentialEvolutionOptimizer
(ADEO.py), explained with SHAP.

Instead of the fixed 4 hand-picked combinations in ADEO.py's
parameter_sensitivity_analysis(), this script:
  1. Randomly samples many (F, CR, adapt_factor, pop_size) combinations
     across CEC2022 functions and dimensions.
  2. Runs ADEO with each combination and records the resulting error.
  3. Trains a RandomForestRegressor to predict error from the parameters.
  4. Uses SHAP (TreeExplainer) to explain which parameters drive
     performance, and how.

Demo settings (N_SAMPLES, MAX_EVALUATIONS) are kept small so this finishes
in a couple of minutes. Increase them for a more reliable analysis once
you've confirmed the pipeline works.
"""
import numpy as np
import pandas as pd

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
N_SAMPLES = 300
MAX_EVALUATIONS = 8000
RUNS_PER_SAMPLE = 2
RANDOM_SEED = 42


class AdaptiveDifferentialEvolutionOptimizer:
    """Same core loop as ADEO.py, parameterized for the sweep."""

    def __init__(self, max_evaluations, dim, pop_size, F, CR, adapt_factor):
        self.max_evaluations = max_evaluations
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.adapt_factor = adapt_factor
        self.bounds = BOUNDS
        self.eval_count = 0

    def optimize(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func.evaluate(ind) for ind in population])
        self.eval_count = self.pop_size

        current_F = self.F
        current_CR = self.CR

        while self.eval_count < self.max_evaluations:
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + current_F * (b - c), self.bounds[0], self.bounds[1])

                cross_points = np.random.rand(self.dim) < current_CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                f_trial = func.evaluate(trial)
                self.eval_count += 1
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial

                if self.eval_count >= self.max_evaluations:
                    break

            current_F *= self.adapt_factor
            current_CR *= self.adapt_factor

        best_idx = np.argmin(fitness)
        return fitness[best_idx]


def sample_parameters(rng, n_samples):
    return pd.DataFrame({
        "F": rng.uniform(0.3, 1.0, n_samples),
        "CR": rng.uniform(0.1, 1.0, n_samples),
        "adapt_factor": rng.uniform(0.90, 1.0, n_samples),
        "pop_size": rng.integers(20, 150, n_samples),
        "dimension": rng.choice([10, 20], n_samples),
        "function_id": rng.integers(1, 13, n_samples),
    })


def run_sweep():
    rng = np.random.default_rng(RANDOM_SEED)
    params_df = sample_parameters(rng, N_SAMPLES)

    records = []
    for row_idx, row in params_df.iterrows():
        func_class = CEC2022_FUNCTIONS[int(row["function_id"]) - 1]
        func = func_class(ndim=int(row["dimension"]))

        errors = []
        for _ in range(RUNS_PER_SAMPLE):
            optimizer = AdaptiveDifferentialEvolutionOptimizer(
                max_evaluations=MAX_EVALUATIONS,
                dim=int(row["dimension"]),
                pop_size=int(row["pop_size"]),
                F=row["F"],
                CR=row["CR"],
                adapt_factor=row["adapt_factor"],
            )
            best_fitness = optimizer.optimize(func)
            errors.append(max(best_fitness - func.f_global, 0.0))

        mean_error = float(np.mean(errors))
        records.append({**row.to_dict(), "mean_error": mean_error, "log_error": np.log1p(mean_error)})

        if (row_idx + 1) % 25 == 0:
            print(f"  sweep progress: {row_idx + 1}/{N_SAMPLES}")

    return pd.DataFrame(records)


def analyze_with_shap(results_df):
    from sklearn.ensemble import RandomForestRegressor
    import shap
    import matplotlib.pyplot as plt

    feature_cols = ["F", "CR", "adapt_factor", "pop_size", "dimension", "function_id"]
    X = results_df[feature_cols]
    y = results_df["log_error"]

    model = RandomForestRegressor(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X, y)
    print(f"Model R^2 on training data: {model.score(X, y):.3f}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("shap_feature_importance_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()

    mean_abs_shap = pd.Series(
        np.abs(shap_values.values).mean(axis=0), index=feature_cols
    ).sort_values(ascending=False)

    print("\nMean |SHAP value| per parameter (higher = bigger influence on error):")
    print(mean_abs_shap.to_string())

    return model, shap_values, mean_abs_shap


def main():
    print(f"Running parameter sweep: {N_SAMPLES} samples x {RUNS_PER_SAMPLE} runs each...")
    results_df = run_sweep()
    results_df.to_csv("parameter_sensitivity_sweep_results.csv", index=False)
    print("Sweep results saved to parameter_sensitivity_sweep_results.csv")

    print("\nTraining surrogate model and computing SHAP values...")
    model, shap_values, mean_abs_shap = analyze_with_shap(results_df)

    mean_abs_shap.to_csv("shap_parameter_importance.csv", header=["mean_abs_shap"])
    print("\nSaved plots: shap_feature_importance_bar.png, shap_summary_beeswarm.png")
    print("Saved ranking: shap_parameter_importance.csv")


if __name__ == "__main__":
    main()

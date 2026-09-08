"""
SHAP Options 2 & 3, run on the combined production_results_<algo>.csv files
produced by ioh_production_runner.py once all 10 algorithms have finished
their 30-run x 12-function x 2-dimension production benchmark.

Option 2 - Algorithm-selection explainability:
    Train a classifier that predicts which of the 10 algorithms is best for
    a given (Function_ID, Dimension), then use SHAP to explain which of
    those two drives the recommendation more.

Option 3 - Function-difficulty / error-driver attribution:
    Train a regressor that predicts log(error) from (Algorithm, Function_ID,
    Dimension), then use SHAP to see whether algorithm choice, function
    landscape, or dimensionality drives error the most across the whole
    10-algorithm benchmark.
"""
import glob

import numpy as np
import pandas as pd


def load_combined_results():
    files = sorted(glob.glob("production_results_*.csv"))
    if not files:
        raise FileNotFoundError(
            "No production_results_*.csv files found yet — "
            "run all 10 ioh_production_runner.py jobs first."
        )
    frames = [pd.read_csv(f) for f in files]
    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(files)} algorithm result files, {len(combined)} total rows.")
    missing = set(f"production_results_{k}.csv" for k in [
        "adeo", "de_crossover", "de_gradient_boost", "de_harmony_search",
        "de_local_search", "de_memetic_search", "de_perturbation",
        "de_plus", "de_pso", "dynamic_pop_v2",
    ]) - set(files)
    if missing:
        print(f"WARNING: still missing {missing} — results below are partial.")
    combined.to_csv("all_algorithms_production_combined.csv", index=False)
    return combined


def option2_algorithm_selection(combined):
    from sklearn.ensemble import RandomForestClassifier
    import shap
    import matplotlib.pyplot as plt

    mean_error = (
        combined.groupby(["Algorithm", "Function_ID", "Dimension"])["Error"]
        .mean()
        .reset_index()
    )
    best_idx = mean_error.groupby(["Function_ID", "Dimension"])["Error"].idxmin()
    best_algo_table = mean_error.loc[best_idx].reset_index(drop=True)
    best_algo_table = best_algo_table.rename(columns={"Algorithm": "Best_Algorithm", "Error": "Best_Mean_Error"})
    best_algo_table.to_csv("best_algorithm_per_function_dimension.csv", index=False)
    print("\n=== Best algorithm per (Function_ID, Dimension) ===")
    print(best_algo_table.to_string(index=False))

    labeled = combined.merge(best_algo_table[["Function_ID", "Dimension", "Best_Algorithm"]],
                              on=["Function_ID", "Dimension"], how="left")

    X = labeled[["Function_ID", "Dimension"]]
    y = labeled["Best_Algorithm"]

    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    print(f"\n[Option 2] Classifier accuracy on training data: {clf.score(X, y):.3f}")

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X)

    values = shap_values.values
    is_multiclass = values.ndim == 3
    mean_abs_per_class = np.abs(values).mean(axis=(0, 2)) if is_multiclass else np.abs(values).mean(axis=0)
    importance = pd.Series(mean_abs_per_class, index=X.columns).sort_values(ascending=False)
    print("\n[Option 2] Mean |SHAP| across all algorithm classes:")
    print(importance.to_string())
    importance.to_csv("shap_option2_algorithm_selection_importance.csv", header=["mean_abs_shap"])

    plt.figure()
    class_idx = 0
    class_values = shap_values[:, :, class_idx] if is_multiclass else shap_values
    shap.summary_plot(class_values, X, show=False)
    plt.title(f"SHAP for predicting best-algorithm = {clf.classes_[class_idx]}")
    plt.tight_layout()
    plt.savefig("shap_option2_algorithm_selection_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()


def option3_function_difficulty(combined):
    from sklearn.ensemble import RandomForestRegressor
    import shap
    import matplotlib.pyplot as plt

    df = combined.copy()
    df["log_error"] = np.log1p(df["Error"].clip(lower=0))
    algo_categories = sorted(df["Algorithm"].unique())
    df["Algorithm_Code"] = df["Algorithm"].astype(pd.CategoricalDtype(categories=algo_categories)).cat.codes

    feature_cols = ["Algorithm_Code", "Function_ID", "Dimension"]
    X = df[feature_cols]
    y = df["log_error"]

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X, y)
    print(f"\n[Option 3] Model R^2 on training data: {model.score(X, y):.3f}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    mean_abs_shap = pd.Series(
        np.abs(shap_values.values).mean(axis=0), index=feature_cols
    ).sort_values(ascending=False)
    print("\n[Option 3] Mean |SHAP| per feature (drives error the most):")
    print(mean_abs_shap.to_string())
    mean_abs_shap.to_csv("shap_option3_function_difficulty_importance.csv", header=["mean_abs_shap"])

    algo_code_map = pd.Series(range(len(algo_categories)), index=algo_categories)
    algo_code_map.to_csv("shap_option3_algorithm_code_map.csv", header=["code"])

    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("shap_option3_function_difficulty_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig("shap_option3_function_difficulty_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()


def main():
    combined = load_combined_results()
    option2_algorithm_selection(combined)
    option3_function_difficulty(combined)
    print("\nDone. See shap_option2_*.png/csv and shap_option3_*.png/csv")


if __name__ == "__main__":
    main()
